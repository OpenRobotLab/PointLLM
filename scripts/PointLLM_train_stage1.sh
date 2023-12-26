master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Get the filename without extension
filename=$(basename "$0" | cut -f 1 -d '.')

dir_path=PointLLM
model_name_or_path=checkpoints/PointLLM_7B_v1.1_init
data_path=data/objaverse_data
anno_path=data/anno_data/PointLLM_brief_description_660K_filtered.json # or PointLLM_brief_description_660K.json (including val sets)
output_dir=outputs/PointLLM_train_stage1/$filename
point_backbone_ckpt=$model_name_or_path/point_bert_v1.2.pt

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
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --evaluation_strategy "no" \
    --fix_llm True \
    --fix_pointnet True \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name $filename \
    --point_backbone_ckpt $point_backbone_ckpt \
    --use_color True