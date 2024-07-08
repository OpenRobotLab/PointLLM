#    Copyright 2023 Runsen Xu

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from .utils import *
from pointllm.utils import *

from contextlib import nullcontext
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import os

# * add logger
import logging
logger = logging.getLogger(__name__)

class PointLLMConfig(LlamaConfig):
    model_type = "pointllm"

class PointLLMLlamaModel(LlamaModel):
    config_class = PointLLMConfig 

    def __init__(self, config: LlamaConfig):
        super(PointLLMLlamaModel, self).__init__(config)

        self.point_backbone_type = config.point_backbone
        logger.info(f"Using {self.point_backbone_type}.")

        if self.point_backbone_type == "PointBERT":
            from pointllm.model import PointTransformer
            # address of config file, in the same dir of this file
            point_bert_config_name = getattr(config, "point_backbone_config_name", "PointTransformer_8192point_2layer") # * default for v1.2, v1.1 uses PointTransformer_base_8192point.yaml
            point_bert_config_addr = os.path.join(os.path.dirname(__file__), "pointbert", f"{point_bert_config_name}.yaml")
            print(f"Loading PointBERT config from {point_bert_config_addr}.")
            point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
            if getattr(config, "use_color", False):
                point_bert_config.model.point_dims = 6
            use_max_pool = getattr(point_bert_config.model, "use_max_pool", False) # * default is false
            
            self.point_backbone = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool)
            logger.info(f"Using {self.point_backbone.point_dims} dim of points.")

            self.point_backbone_config = {
                "point_cloud_dim": point_bert_config.model.point_dims,
                "backbone_output_dim": point_bert_config.model.trans_dim if not use_max_pool else point_bert_config.model.trans_dim * 2,
                "project_output_dim": self.config.hidden_size,
                "point_token_len": point_bert_config.model.num_group + 1 if not use_max_pool else 1, # * number of output features, with cls token
                "mm_use_point_start_end": self.config.mm_use_point_start_end,
                "projection_hidden_layer": point_bert_config.model.get('projection_hidden_layer', 0),
                "use_max_pool": use_max_pool
            }
            if point_bert_config.model.get('projection_hidden_layer', 0) > 0:
                self.point_backbone_config["projection_hidden_dim"] = point_bert_config.model.projection_hidden_dim # a list
            
            logger.info(f"Use max pool is {use_max_pool}. Number of point token is {self.point_backbone_config['point_token_len']}.")

        # * print relevant info with projection layers
        backbone_output_dim = self.point_backbone_config["backbone_output_dim"]
        logger.info(f"Point backbone output dim: {backbone_output_dim}.")
        logger.info(f"Use {self.point_backbone_config['projection_hidden_layer']} projection hiddent layers.")
        if self.point_backbone_config['projection_hidden_layer'] > 0:
            # Add projection layer with linear layers and GELU activation
            projection_layers = []
            last_dim = backbone_output_dim
            for i in range(point_bert_config.model.projection_hidden_layer):
                projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["projection_hidden_dim"][i]))
                projection_layers.append(nn.GELU())
                last_dim = self.point_backbone_config["projection_hidden_dim"][i]

            projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["project_output_dim"]))
            self.point_proj = nn.Sequential(*projection_layers)
            logger.info(f"Each layer with {point_bert_config.model.projection_hidden_dim} hidden units.")
        else:
            # Single layer
            self.point_proj = nn.Linear(backbone_output_dim, self.point_backbone_config['project_output_dim'])
        logger.info(f"Point projector output dim: {self.point_backbone_config['project_output_dim']}.")

        self.fix_pointnet = False
        self.fix_llm = False

    def load_point_backbone_checkpoint(self, checkpoint_path=None):
        self.point_backbone.load_checkpoint(self.config.point_backbone_ckpt if checkpoint_path is None else checkpoint_path)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        point_backbone = getattr(self, 'point_backbone', None)
        point_backbone_config = getattr(self, 'point_backbone_config', None)

        if point_backbone is not None and (input_ids.shape[1] != 1 or self.training) and point_clouds is not None:
            # * enter when training or the first generation step of inference
            with torch.no_grad() if self.fix_pointnet else nullcontext():
                if self.fix_pointnet:
                    self.point_backbone.eval()
                if type(point_clouds) is list:
                    # * variable numbers of points
                    point_features = []
                    for point_cloud in point_clouds: # * iterate over batch
                        point_feature = self.point_backbone(point_cloud.unsqueeze(0))[0]
                        point_features.append(point_feature)
                else:
                    point_features = self.point_backbone(point_clouds)

            if type(point_clouds) is list:
                point_features = [self.point_proj(point_feature) for point_feature in point_features]
            else:
                point_features = self.point_proj(point_features)

            dummy_point_features = torch.zeros(point_backbone_config['point_token_len'], point_backbone_config['backbone_output_dim'], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_point_features = self.point_proj(dummy_point_features)

            new_input_embeds = []
            cur_point_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds): # * input_ids: B, L; input_embeds: B, L, C
                if (cur_input_ids == point_backbone_config['point_patch_token']).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_point_features).sum() # * do nothing
                    new_input_embeds.append(cur_input_embeds)
                    cur_point_idx += 1
                    continue
                cur_point_features = point_features[cur_point_idx].to(device=cur_input_embeds.device)
                num_patches = cur_point_features.shape[0] # * number of point tokens
                if point_backbone_config['mm_use_point_start_end']:
                    if (cur_input_ids == point_backbone_config["point_start_token"]).sum() != (cur_input_ids == point_backbone_config["point_end_token"]).sum():
                        raise ValueError("The number of point start tokens and point end tokens should be the same.")
                    point_start_tokens = torch.where(cur_input_ids == point_backbone_config["point_start_token"])[0]
                    for point_start_token_pos in point_start_tokens:
                        if cur_input_ids[point_start_token_pos + num_patches + 1] != point_backbone_config["point_end_token"]:
                            raise ValueError("The point end token should follow the point start token.")
                        if orig_embeds_params is not None: # * will not update the original embeddings except for POINT_START_TOKEN and POINT_END_TOKEN
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos].detach(), cur_input_embeds[point_start_token_pos:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:point_start_token_pos + num_patches + 2], cur_input_embeds[point_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:point_start_token_pos+1], cur_point_features, cur_input_embeds[point_start_token_pos + num_patches + 1:]), dim=0)
                        cur_point_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    if (cur_input_ids == point_backbone_config["point_patch_token"]).sum() != num_patches:
                        raise ValueError("The number of point patch tokens should be the same as the number of point patches.")
                    masked_indices = torch.where(cur_input_ids == point_backbone_config["point_patch_token"])[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The point patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_point_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_point_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_point_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(PointLLMLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class PointLLMLlamaForCausalLM(LlamaForCausalLM):
    config_class = PointLLMConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = PointLLMLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None, # * control whether to return past_key_values
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            point_clouds=point_clouds
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous() # * B, L, V(32003)
            shift_labels = labels[..., 1:].contiguous() # * B, L
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "point_clouds": kwargs.get("point_clouds", None),
            }
        )
        return model_inputs

    def initialize_tokenizer_point_backbone_config_wo_embedding(self, tokenizer):
        # * called when stage2 or inference or inference without pre-training, assume tokenizer has point tokens
        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN

        tokenizer.add_tokens([default_point_patch_token], special_tokens=True)

        # * assert tokenizer has the default_point_patch_token
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)

            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token

            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]
    
    def initialize_tokenizer_point_backbone_config(self, tokenizer, device, fix_llm=True):

        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        tokenizer.add_tokens([default_point_patch_token], special_tokens=True) # * no need to update embed since it will be replaced
        self.resize_token_embeddings(len(tokenizer)) # ! resize_token_embeddings will make the tokens trainable again
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token

            num_new_tokens = tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                # need to update the input embeding, but no need to update the output embedding
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                if fix_llm:
                    self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)] # * only tuning the new embeddings
                    for p in self.get_output_embeddings().parameters(): # * the llm head
                        p.requires_grad = False
                    print(f"Setting output embeddings fixed and {num_new_tokens} new tokens' input embeddings trainable.")
                else:
                    self.get_model().orig_embeds_params = None
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = True
                    print("Setting output embeddings and all input embeddings trainable.")

AutoConfig.register("pointllm", PointLLMConfig)
AutoModelForCausalLM.register(PointLLMConfig, PointLLMLlamaForCausalLM)
