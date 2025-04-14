PROMPT_VERSION="qwen_1_5"
LLM_VERSION="/path_to_your_dir/Qwen2.5-0.5B-Instruct" 

IFS=',' read -ra ALL_PORTS <<< $METIS_WORKER_0_PORT
export NCCL_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ACCELERATE_CPU_AFFINITY=1 \
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --gen_data_path /path_to_annotation_file \
    --gen_image_folder "" \
    --token_dataset True \
    --sample_short True \
    --mm_tunable_parts="mm_language_model" \
    --p_drop_cond 0.1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name test \
    --output_dir /path_to_output_dir \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --report_to wandb \
    --attn_implementation sdpa