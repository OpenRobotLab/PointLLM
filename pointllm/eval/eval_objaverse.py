import argparse
import torch
from torch.utils.data import DataLoader
import os
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import *
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.data import ObjectPointCloudDataset
from tqdm import tqdm
from transformers import AutoTokenizer
from pointllm.eval.evaluator import start_evaluation

import os
import json

PROMPT_LISTS = [
    "What is this?",
    "This is an object of ",
    "Caption this 3D model in detail."
]

def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.bfloat16).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"

    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv

def load_dataset(data_path, anno_path, pointnum, conversation_types, use_color):
    print("Loading validation datasets.")
    dataset = ObjectPointCloudDataset(
        data_path=data_path,
        anno_path=anno_path,
        pointnum=pointnum,
        conversation_types=conversation_types,
        use_color=use_color,
        tokenizer=None # * load point cloud only
    )
    print("Done!")
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    model.eval() 
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            stopping_criteria=[stopping_criteria]) # * B, L'

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    return outputs

def start_generation(model, tokenizer, conv, dataloader, annos, prompt_index, output_dir, output_file):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = PROMPT_LISTS[prompt_index]

    results = {"prompt": qs}

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']

    if mm_use_point_start_end:
        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
    else:
        qs = default_point_patch_token * point_token_len + '\n' + qs
    
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])

    input_ids_ = torch.as_tensor(inputs.input_ids).cuda() # * tensor of 1, L

    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    responses = []

    for batch in tqdm(dataloader):
        point_clouds = batch["point_clouds"].cuda().to(model.dtype) # * tensor of B, N, C(3)
        object_ids = batch["object_ids"] # * list of string 

        batchsize = len(object_ids)

        input_ids = input_ids_.repeat(batchsize, 1) # * tensor of B, L

        outputs = generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria) # List of str, length is B

        # saving results
        for obj_id, output in zip(object_ids, outputs):
            responses.append({
                "object_id": obj_id,
                "ground_truth": annos[obj_id],
                "model_output": output
            })
    
    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results

def main(args):
    # * ouptut
    args.output_dir = os.path.join(args.model_name, "evaluation")
    
    # * output file 
    anno_file = os.path.splitext(os.path.basename(args.anno_path))[0]
    args.output_file = f"{anno_file}_Objaverse_{args.task_type}_prompt{args.prompt_index}.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    # * First inferencing, then evaluate
    if not os.path.exists(args.output_file_path):
        # * need inferencing
        # * load annotation files
        with open(args.anno_path, 'r') as fp:
            annos = json.load(fp)

        dataset = load_dataset(args.data_path, args.anno_path, args.pointnum, ("simple_description",), args.use_color)
        dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)
        
        model, tokenizer, conv = init_model(args)

        # * convert annos file from [{"object_id": }] to {"object_id": }
        annos = {anno["object_id"]: anno["conversations"][1]['value'] for anno in annos}

        print(f'[INFO] Start generating results for {args.output_file}.')
        results = start_generation(model, tokenizer, conv, dataloader, annos, args.prompt_index, args.output_dir, args.output_file)

        # * release model and tokenizer, and release cuda memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        # * directly load the results
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    if args.start_eval:
        evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
        eval_type_mapping = {
            "captioning": "object-captioning",
            "classification": "open-free-form-classification"
        }
        start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file, eval_type=eval_type_mapping[args.task_type], model_type=args.gpt_type, parallel=True, num_workers=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, \
        default="RunsenXu/PointLLM_7B_v1.2") 

    # * dataset type
    parser.add_argument("--data_path", type=str, default="data/objaverse_data", required=False)
    parser.add_argument("--anno_path", type=str, default="data/anno_data/PointLLM_brief_description_val_200_GT.json", required=False)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--use_color",  action="store_true", default=True)

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=10)

    # * evaluation setting
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-4-0613", choices=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview"], help="Type of the model used to evaluate.")
    parser.add_argument("--task_type", type=str, default="captioning", choices=["captioning", "classification"], help="Type of the task to evaluate.")

    args = parser.parse_args()

    # * check prompt index
    # * * classification: 0, 1 and captioning: 2. Raise Warning otherwise.
    if args.task_type == "classification":
        if args.prompt_index != 0 and args.prompt_index != 1:
            print("[Warning] For classification task, prompt_index should be 0 or 1.")
    elif args.task_type == "captioning":
        if args.prompt_index != 2:
            print("[Warning] For captioning task, prompt_index should be 2.")
    else:
        raise NotImplementedError

    main(args)