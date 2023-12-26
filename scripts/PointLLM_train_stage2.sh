master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
filename=$(basename "$0" | cut -f 1 -d '.')

dir_path=PointLLM

model_name_or_path=outputs/PointLLM_train_stage1/PointLLM_train_stage1 # Path to the output dir of stage 1 training
data_path=data/objaverse_data
anno_path=data/anno_data/PointLLM_complex_instruction_70K.json
output_dir=outputs/PointLLM_train_stage2/$filename

cd $dir_path

PYTHONPATH=$dir_path:$PYTHONPATH \
torchrun --nnodes=1 --nproc_per_node=8 --master_port=$master_port pointllm/train/train_mem.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --anno_path $anno_path \
    --output_dir $output_dir \
    --version v1 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 100 \
    --save_strategy "no" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --fix_llm False \
    --fix_pointnet True \
    --report_to wandb \
    --run_name $filename \
    --gradient_checkpointing True \
    --stage_2 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --conversation_types "detailed_description" "single_round" "multi_round" \
    --use_color True