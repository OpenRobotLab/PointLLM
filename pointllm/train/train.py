#  Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import pathlib
from typing import Optional, List


import transformers
from pointllm.train.pointllm_trainer import PointLLMTrainer

from pointllm import conversation as conversation_lib
from pointllm.model import *
from pointllm.data import make_object_point_data_module

# * logger
from pointllm.utils import build_logger

IGNORE_INDEX = -100

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    version: Optional[str] = field(default="v1")

@dataclass
class DataArguments:
    data_path: str = field(default="ScanNet", metadata={"help": "Path to the training data."})
    anno_path: str = field(default=None, metadata={"help": "Path to the utterance data. If None, will use referit3d by defautl."})
    use_color: bool = field(default=False, metadata={"help": "Whether to use color."})
    data_debug_num: int = field(default=0, metadata={"help": "Number of data to use in debug mode. If larger than 0, use debug mode, else use the whole data"})
    split_train_val: bool = field(default=False, metadata={"help": "Whether to split train and val."})
    split_ratio: float = field(default=0.9, metadata={"help": "Ratio of train and val."})
    pointnum: int = field(default=8192, metadata={"help": "Number of points."})
    conversation_types: List[str] = field(default_factory=lambda: ["simple_description"], metadata={"help": "Conversation types to use."})
    is_multimodal: bool = True

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # * can refer to https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/trainer#transformers.TrainingArgument
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    model_debug: bool = field(default=False, metadata={"help": "Whether to use small model."}) # * whether to load checkpoints at the mo
    fix_llm: bool = field(default=True, metadata={"help": "Whether to fix the LLM."})
    fix_pointnet: bool = field(default=True, metadata={"help": "Whether to fix the PointNet."})

    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)

    # * for two stage training
    tune_mm_mlp_adapter: bool = field(default=True) # * set True when pre-training, and false when fine-tuning
    stage_2: bool = field(default=False) # * set True when fine-tuning
    pretrained_mm_mlp_adapter: Optional[str] = field(default=None) # * path to the pre-trained projector & output_embed & input_embed
    detatch_point_token: bool = field(default=False) # * deprecated
    # * point backbone ckpt path
    point_backbone_ckpt: str = field(default=None)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.log_level = "info" # * default is passive(warning)
    # * build logger
    logger = build_logger(__name__, training_args.output_dir + '/train.log')

    if training_args.model_debug:
        # * do not load checkpoint, load from config
        config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )
        model = PointLLMLlamaForCausalLM._from_config(config)
    else:
        model = PointLLMLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )

    model.config.use_cache = False

    if training_args.fix_llm:
        # * This will fix all the parameters
        logger.info("LLM is fixed. Fix_llm flag is set to True")
        # * fix llama, lm_head, pointnet, projection layer here
        model.requires_grad_(False)
        model.get_model().fix_llm = True
        model.get_model().point_proj.requires_grad_(True) 
        model.get_model().point_backbone.requires_grad_(True) # * set as True for fsdp, use fix_pointnet flag to control
    else:
        model.get_model().fix_llm = False
        logger.warning("LLM is trainable. Fix_llm flag is set to False")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0" or "v0" in model_args.model_name_or_path:
        raise ValueError("v0 is deprecated.")
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    if not training_args.fix_pointnet:
        # * not fix pointnet
        logger.info("Point backbone is trainable. Fix_pointnet flag is set to False, pointnet grad will be recorded.")
        model.get_model().fix_pointnet = False
    else:
        logger.info("Point backbone is fixed. Fix_pointnet flag is set to True, pointnet grad will not be recorded.")
        model.get_model().fix_pointnet = True # * use with torch.inference_mode to control, not requires_grad for fsdp for second stage
        if not training_args.stage_2:
            logger.info("Set requires_grad of point backbone to False")
            model.get_model().point_backbone.requires_grad_(False) # * fix pointnet for first stage, need for fsdp in stage2
    
    if training_args.tune_mm_mlp_adapter:
        # * not fix the projection layer
        # * may need to set the embed_tokens to require_grad = True if added new tokens
        # * this is done in initialize_tokenizer_point_backbone_config
        logger.info("Point projection layer is trainable.")
    else:
        model.get_model().point_proj.requires_grad_(False)
        logger.info("Point prejcetion layer is fixed.")

    if not training_args.stage_2:
        # * we assume in stage2, llm, point_backbone, and projection layer can be loaded from the model checkpoint
        print(f"Default point_backbone_ckpt is {training_args.point_backbone_ckpt}.")
        model.get_model().load_point_backbone_checkpoint(training_args.point_backbone_ckpt)
        model.initialize_tokenizer_point_backbone_config(tokenizer=tokenizer, device=training_args.device, fix_llm=training_args.fix_llm)
    else:
        # * stage2
        model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer=tokenizer) 

    point_backbone_config = model.get_model().point_backbone_config

    data_args.point_token_len = point_backbone_config['point_token_len']
    data_args.mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']
    data_args.point_backbone_config = point_backbone_config

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)
                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    data_module = make_object_point_data_module(tokenizer=tokenizer,
                                                    data_args=data_args)

    trainer = PointLLMTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
