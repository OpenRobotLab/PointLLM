import argparse
from transformers import AutoTokenizer
import torch
import os
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import *
from pointllm.model.utils import KeywordsStoppingCriteria
import numpy as np

from pointllm.data import pc_norm, farthest_point_sample

import os

# Additional import for gradio
import gradio as gr
import open3d as o3d
import plotly.graph_objects as go
import objaverse
import time

import logging


def change_input_method(input_method):
    if input_method == 'File':
        result = [gr.update(visible=True),
        gr.update(visible=False)]
    elif input_method == 'Object ID':
        result = [gr.update(visible=False),
        gr.update(visible=True)]
    return result

def init_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')
    logging.warning(f'Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    model.eval()

    mm_use_point_start_end = getattr(model.config, "mm_use_point_start_end", False)
    # Add special tokens ind to model.point_config
    point_backbone_config = model.get_model().point_backbone_config
    
    conv = conv_templates["vicuna_v1_1"].copy()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    
    return model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv

def start_conversation(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv):
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']

    # The while loop will keep running until the user decides to quit
    print("[INFO] Starting conversation...")
    logging.warning("Starting conversation...")
    while True:
        print("-" * 80)
        logging.warning("-" * 80)

        # Reset the conversation template
        conv.reset()

        def confirm_point_cloud(input_choice, object_id_input, point_cloud_input, chatbot, answer_time, conv):
            objects = None
            data = None
            object_id_input = object_id_input.strip()

            print("%" * 80)
            logging.warning("%" * 80)

            if input_choice == 'File':
                file = point_cloud_input.name 
                print(f"Uploading file: {file}.")
                logging.warning(f"Uploading file: {file}.")
            elif input_choice == 'Object ID':
                file = os.path.join(args.data_path, "{}_8192.npy".format(object_id_input))
                print(f"Object_id: {object_id_input}")
                logging.warning(f"Object_id: {object_id_input}")

                object_uids = [object_id_input]
                objects = objaverse.load_objects(uids=object_uids)
            print("%" * 80)
            logging.warning("%" * 80)

            manual_no_color = "no_color" in file

            try:
                if '.ply' in file:
                    pcd = o3d.io.read_point_cloud(file)
                    points = np.asarray(pcd.points)  # xyz
                    colors = np.asarray(pcd.colors)  # rgb, if available
                    # * if no colors actually, empty array
                    if colors.size == 0:
                        colors = None
                elif '.npy' in file:
                    data = np.load(file)
                    if data.shape[1] >= 3:
                        points = data[:, :3]
                    else:
                        raise ValueError("Input array has the wrong shape. Expected: [N, 3]. Got: {}.".format(data.shape))
                    colors = None if data.shape[1] < 6 else data[:, 3:6]
                else:
                    raise ValueError("Not supported data format.")
            # error
            except Exception as e:
                print(f"[ERROR] {e}")
                logging.warning(f"[ERROR] {e}")

                chatbot_system_message = "Sorry. The Objaverse id is not supported or the uploaded file has something wrong!"
                print(f"[ChatBot System Message]: {chatbot_system_message}")
                logging.warning(f"[ChatBot System Message]: {chatbot_system_message}")

                outputs = f"<span style='color: red;'>[System] {chatbot_system_message}</span>" # "You upload a new Points Cloud"
                chatbot = chatbot + [[None, outputs]]
            
                return None, None, chatbot, answer_time, None

            if manual_no_color:
                colors = None

            if colors is not None:
                # * if colors in range(0-1)
                if np.max(colors) <= 1:
                    color_data = np.multiply(colors, 255).astype(int)  # Convert float values (0-1) to integers (0-255)
                # * if colors in range(0-255)
                elif np.max(colors) <= 255:
                    color_data = colors.astype(int)
            else:
                color_data = np.zeros_like(points).astype(int)  # Default to black color if RGB information is not available
            colors = color_data.astype(np.float32) / 255 # model input is (0-1)

            # Convert the RGB color data to a list of RGB strings in the format 'rgb(r, g, b)'
            color_strings = ['rgb({},{},{})'.format(r, g, b) for r, g, b in color_data]

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=points[:, 0], y=points[:, 1], z=points[:, 2],
                        mode='markers',
                        marker=dict(
                            size=1.2,
                            color=color_strings,  # Use the list of RGB strings for the marker colors
                        )
                    )
                ],
                layout=dict(
                    scene=dict(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        zaxis=dict(visible=False)
                    ),
                    paper_bgcolor='rgb(255,255,255)'  # Set the background color to dark gray 50, 50, 50
                ),
            )

            points = np.concatenate((points, colors), axis=1)
            if 8192 < points.shape[0]:
                points = farthest_point_sample(points, 8192)
            point_clouds = pc_norm(points)
            point_clouds = torch.from_numpy(point_clouds).unsqueeze_(0).to(torch.float32).cuda()
            
            answer_time = 0
            conv.reset()
            
            outputs = "<span style='color: red;'>[System] New Point Cloud</span>" 
            chatbot = chatbot + [[None, outputs]]
            
            return fig, list(objects.values())[0] if objects is not None else None, chatbot, answer_time, point_clouds

        def answer_generate(history, answer_time, point_clouds, conv):
            if point_clouds is None:
                outputs = "<span style='color: red;'>[System] Please input point cloud! </span>"
                history[-1][1] = outputs
                yield history
            else:            
                print(f"Answer Time: {answer_time}")
                logging.warning(f"Answer Time: {answer_time}")
                input_text = history[-1][0]
                qs = input_text
                
                if answer_time == 0:
                    if mm_use_point_start_end:
                        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
                    else:
                        qs = default_point_patch_token * point_token_len + '\n' + qs

                # Append the new message to the conversation history
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                print("#" * 80)
                print(f'{prompt.replace("<point_patch>" * point_token_len, f"<point_patch> * {point_token_len}")}') # for concise printing
                print("#" * 80)

                logging.warning("#" * 80)
                logging.warning(f'{prompt.replace("<point_patch>" * point_token_len, f"<point_patch> * {point_token_len}")}') # for concise printing
                logging.warning("#" * 80)
                inputs = tokenizer([prompt])

                input_ids = torch.as_tensor(inputs.input_ids).cuda()

                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                stop_str = keywords[0]

                try:
                    if input_ids.shape[1] >= 2047:
                        raise ValueError("Current context length exceeds the maximum context length (2048) of the model.")
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
                        logging.warning(f'{n_diff_input_output} output_ids are not the same as the input_ids')
                    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                    outputs = outputs.strip()
                    if outputs.endswith(stop_str):
                        outputs = outputs[:-len(stop_str)]
                    outputs = outputs.strip()

                    # Append the model's response to the conversation history
                    conv.pop_last_none_message()
                    conv.append_message(conv.roles[1], outputs)
                    print(f'{conv.roles[1]}: {outputs}\n')
                    logging.warning(f'{conv.roles[1]}: {outputs}\n')
                    answer_time += 1
                    history[-1][1] = ""
                    for character in outputs:
                        history[-1][1] += character
                        yield history
                # error
                except Exception as e:
                    print(f"[ERROR] {e}")
                    logging.warning(f"[ERROR] {e}")

                    if input_ids.shape[1] >= 2047:
                        chatbot_system_message = "Current context length exceeds the maximum context length (2048) of the model. Please press 'Clear' to restart."
                    else:
                        chatbot_system_message = "Sorry. There is something wrong when generating. Please check the your uploaded point cloud or the Objaverse id, and \
                        confirm the point cloud again."
                    print(f"[ChatBot System Message]: {chatbot_system_message}")
                    logging.warning(f"[ChatBot System Message]: {chatbot_system_message}")

                    outputs = f"<span style='color: red;'>[System] {chatbot_system_message}</span>" # "You upload a new Points Cloud"
                    history[-1][1] = outputs
                    yield history
            
        with gr.Blocks() as demo:
            answer_time = gr.State(value=0)
            point_clouds = gr.State(value=None)
            conv_state = gr.State(value=conv.copy())
            gr.Markdown(
                """
                # PointLLM: Empowering Large Language Models to Understand Point Clouds. 🚀
                If you think this demo interesting, please consider starring 🌟 our github repo. :)
                [[Project Page](https://runsenxu.com/projects/PointLLM)] [[Paper](https://arxiv.org/abs/2308.16911)] [[Code](https://github.com/OpenRobotLab/PointLLM)] 
                """
            )
            with gr.Row():
                with gr.Column():
                    input_choice = gr.Radio(['File', 'Object ID'], value='Object ID', interactive=True, label='Input Method', info="How do you want to load point clouds?")
                    object_id_input = gr.Textbox(visible = True,lines=1, label='Object ID Input')
                    point_cloud_input = gr.File(visible = False, label="Upload Point Cloud File (PLY, NPY)")
                    output = gr.Plot()
                    btn = gr.Button(value="Confirm Point Cloud")
                model3D = gr.Model3D()
                with gr.Column():
                    chatbot  = gr.Chatbot([], elem_id="chatbot", height=560) # ,color_map=("green", "pink")

                    def user(user_message, history):
                        return "", history + [[user_message, None]]
                    
                    def clear_conv(history, conv):
                        conv.reset()
                        return None, 0

                    with gr.Row():
                        text_input = gr.Textbox(
                                show_label=False,
                                placeholder="Enter text and press enter",
                                container=False,
                            )
                        run_button = gr.Button("Send")           

                    clear = gr.Button("Clear")
                    text_input.submit(user, [text_input, chatbot], [text_input, chatbot], queue=False).then(answer_generate, [chatbot, answer_time, point_clouds, conv_state], chatbot).then(lambda x : x+1,answer_time, answer_time)
                    clear.click(clear_conv, inputs=[chatbot, conv_state], outputs=[chatbot, answer_time], queue=False)

                btn.click(confirm_point_cloud, inputs=[input_choice, object_id_input, point_cloud_input, chatbot, answer_time, conv_state], outputs=[output, model3D, chatbot, answer_time, point_clouds])
                    
            input_choice.change(change_input_method, input_choice, [point_cloud_input, object_id_input])
            run_button.click(user, [text_input, chatbot], [text_input, chatbot], queue=False).then(answer_generate, [chatbot, answer_time, point_clouds, conv_state], chatbot).then(lambda x : x+1, answer_time, answer_time)

            gr.Markdown(
                """
                ### Usage:
                1. Upload your point cloud file (ply, npy only) or input the supported [Objaverse object id (uid)](https://drive.google.com/file/d/1gLwA7aHfy1KCrGeXlhICG9rT2387tWY8/view?usp=sharing) (currently 660K objects only, you may try the example object ids below). 
                2. If your point cloud file does not contian colors, manually set the file name contains 'no_color' (e.g., 'xxx_no_color.npy'), and the black color will be assigned.
                3. If uploading your own point cloud file with color in npy format, the first three dimensions should be xyz, and the next three dimensions should be rgb. The rgb values should range from **0 to 1**.
                4. Click **Confirm Point Cloud**.
                5. As we use FPS sampling to downsample the point cloud to 8192 points, it may take a long time to confirm the point cloud if the point cloud has too many points. You may use random sampling to downsample the point cloud before uploading.
                6. Once '[System] New Point Cloud' appears in the dialogue box, a new conversation with PointLLM is initialized.
                7. The 'Clear' button will clear the conversation history.
                """)
            with gr.Accordion("Example Objaverse object ids in the validation set!", open=False):
                example_object_ids = [  ["b4bbf2116b1a41a5a3b9d3622b07074c", "0b8da82a3d7a436f9b585436c4b72f56", "650c53d68d374c18886aab91bcf8bb54"],
                                        ["983fa8b23a084f5dacd157e6c9ceba97", "8fe23dd4bf8542b49c3a574b33e377c3", "83cb2a9e9afb47cd9f45461613796645"],
                                        ["3d679a3888c548afb8cf889915af7fd2", "7bcf8626eaca40e592ffd0aed08aa30b", "69865c89fc7344be8ed5c1a54dbddc20"],
                                        ["252f3b3f5cd64698826fc1ab42614677", "e85ebb729b02402bbe3b917e1196f8d3", "97367c4740f64935b7a5e34ae1398035"],
                                        ["fc8dd5a2fc9f4dd19ad6a64a8a6e89e9", "8257772b0e2f408ba269264855dfea00", "d6a3520486bb474f9b5e72eda8408974"],
                                        ["3d10918e6a9a4ad395a7280c022ad2b9", "00002bcb84af4a4781174e62619f14e2", "76ba80230d454de996878c2763fe7e5c"]]
                gr.DataFrame(
                    type="array",
                    headers=["Example Object IDs"] * 3,
                    row_count=6,
                    col_count=3,
                    value=example_object_ids
                )
            gr.Markdown(
                """
                #### Terms of use
                By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
                """
            )
        demo.queue()
        demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)    # server_port=7832, share=True
    
if __name__ == "__main__":
    # ! To release this demo in public, make sure to start in a place where no important data is stored.
    # ! Please check 1. the lanuch dir 2. the tmp dir (GRADIO_TEMP_DIR)
    # ! refer to https://www.gradio.app/guides/sharing-your-app#security-and-file-access
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, \
         default="RunsenXu/PointLLM_7B_v1.2")


    parser.add_argument("--data_path", type=str, default="data/objaverse_data", required=False)
    parser.add_argument("--pointnum", type=int, default=8192)

    parser.add_argument("--log_file", type=str, default="serving_workdirs/serving_log.txt")
    parser.add_argument("--tmp_dir", type=str, default="serving_workdirs/tmp")

    # For gradio
    parser.add_argument("--port", type=int, default=7810)

    args = parser.parse_args()
    
    # * make serving dirs
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    
    # * add the current time for log name
    args.log_file = args.log_file.replace(".txt", f"_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.txt")

    logging.basicConfig(
        filename=args.log_file, 
        level=logging.WARNING, # * default gradio is info, so use warning
        format='%(asctime)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.warning("-----New Run-----")
    logging.warning(f"args: {args}")

    print("-----New Run-----")
    print(f"[INFO] Args: {args}")

    # * set env variable GRADIO_TEMP_DIR to args.tmp_dir
    os.environ["GRADIO_TEMP_DIR"] = args.tmp_dir

    model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv = init_model(args)
    start_conversation(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv)
