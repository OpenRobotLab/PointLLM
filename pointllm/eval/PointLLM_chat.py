import argparse
from transformers import AutoTokenizer
import torch
import os
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import *
from pointllm.model.utils import KeywordsStoppingCriteria

from pointllm.data import load_ulip2_objaverse_point_cloud

import os

def load_point_cloud(args):
    object_id = args.object_id
    print(f"[INFO] Loading point clouds using object_id: {object_id}")
    point_cloud = load_ulip2_objaverse_point_cloud(args.data_path, object_id, pointnum=8192, use_color=True)
    
    return object_id, torch.from_numpy(point_cloud).unsqueeze_(0).to(torch.float32)

def init_model(args):
    # Model
    disable_torch_init()

    model_path = args.model_path 
    print(f'[INFO] Model name: {model_path}')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False, use_cache=True, torch_dtype=args.torch_dtype).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    model.eval()

    mm_use_point_start_end = getattr(model.config, "mm_use_point_start_end", False)
    # Add special tokens ind to model.point_config
    point_backbone_config = model.get_model().point_backbone_config
    
    if mm_use_point_start_end:
        if "v1" in model_path.lower():
            conv_mode = "vicuna_v1_1"
        else:
            raise NotImplementedError

        conv = conv_templates[conv_mode].copy()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    
    return model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv

def start_conversation(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv):
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    # The while loop will keep running until the user decides to quit
    print("[INFO] Starting conversation... Enter 'q' to exit the program and enter 'exit' to exit the current conversation.")
    while True:
        print("-" * 80)
        # Prompt for object_id
        object_id = input("[INFO] Please enter the object_id or 'q' to quit: ")
        
        # Check if the user wants to quit
        if object_id.lower() == 'q':
            print("[INFO] Quitting...")
            break
        else:
            # print info
            print(f"[INFO] Chatting with object_id: {object_id}.")
        
        # Update args with new object_id
        args.object_id = object_id.strip()
        
        # Load the point cloud data
        try:
            id, point_clouds = load_point_cloud(args)
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
        point_clouds = point_clouds.cuda().to(args.torch_dtype)

        # Reset the conversation template
        conv.reset()

        print("-" * 80)

        # Start a loop for multiple rounds of dialogue
        for i in range(100):
            # This if-else block ensures the initial question from the user is included in the conversation
            qs = input(conv.roles[0] + ': ')
            if qs == 'exit':
                break
            
            if i == 0:
                if mm_use_point_start_end:
                    qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
                else:
                    qs = default_point_patch_token * point_token_len + '\n' + qs

            # Append the new message to the conversation history
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            stop_str = keywords[0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    point_clouds=point_clouds,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    max_length=2048,
                    top_p=0.95,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            # Append the model's response to the conversation history
            conv.pop_last_none_message()
            conv.append_message(conv.roles[1], outputs)
            print(f'{conv.roles[1]}: {outputs}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, \
       default="RunsenXu/PointLLM_7B_v1.1")

    parser.add_argument("--data-path", type=str, default="objaverse_data")
    parser.add_argument("--torch-dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])

    args = parser.parse_args()

    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    args.torch_dtype = dtype_mapping[args.torch_dtype]

    model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv = init_model(args)
    
    start_conversation(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv)