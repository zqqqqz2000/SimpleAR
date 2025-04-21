# Pretrain and SFT

For pretraining and SFT, please follow the instructions below to install the environment:

```bash
python3 -m venv env

source env/bin/activate

pip install -e ".[train]"
```

## Data Preparation

We cache the visual tokens for efficient training. Below is the command to extract visual tokens with Cosmos Tokenizer:

```bash
torchrun \
--nnodes=1 --nproc_per_node=8 \
simpar/data/extract_token.py \
    --dataset_type "image" \
    --dataset_name "example" \
    --code_path "/path_to_saved_tokens" \
    --gen_data_path "/path_to_meta_json" \
    --gen_resolution 1024
```

You can specify the meta data file with *--gen_data_path*, which should be a json file with the following format:

```
{
  "image_path": "path_to_image",
  "caption": "a photo of a cat"
}
```

After this, you can use *./scripts/tokens/generate_meta.py* to prepare a meta file.

## Launch Training

For both pretraining and SFT, we use the following command to train the model:

```bash
ACCELERATE_CPU_AFFINITY=1 \
torchrun \
    --nnodes=4 \
    --nproc_per_node=8 \
    simpar/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path "/path_to_your_dir/Qwen2.5-0.5B-Instruct" \
    --version "qwen_1_5" \
    --gen_data_path /path_to_annotation_file \
    --gen_image_folder "" \
    --sample_short True \
    --mm_tunable_parts="mm_language_model" \
    --p_drop_cond 0.1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name test \
    --output_dir /path_to_output_dir \
    --num_train_epochs 1 \
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
    --model_max_length 1536 \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --report_to wandb \
    --attn_implementation sdpa
```

We set *--model_max_length* to # of visual tokens + 512, i.e., 1536 for 512 pretraining and 4608 for 1024 SFT.

# GRPO Training

We strongly recommend you to maintain different python environments for pretraining/SFT and GRPO training, using venv or conda:

```bash

python3 -m venv env_rl

source env_rl/bin/activate

pip install -e ".[train]"

pip install vllm==0.7.2 # important!

pip install wheel
pip install flash-attn --no-build-isolation

pip install "transformers@git+https://github.com/huggingface/transformers.git@7bbc62474391aff64f63fcc064c975752d1fa4de"

git clone https://github.com/huggingface/trl

cd trl
git reset --hard 69ad852e5654a77f1695eb4c608906fe0c7e8624 # specify the commit id!
pip install -e .
cd ..

mv trl trl_arxiv

mv trl_arxiv/trl ./

rm -rf trl_arxiv

pip uninstall bitsandbytes -y
pip install outlines==0.0.46
pip install latex2sympy2_extended math_verify

pip install clint

sudo apt-get install python3-tk -y
```

We follow [Open-R1](https://github.com/huggingface/open-r1) to implement GRPO training with [trl](https://github.com/huggingface/trl), please first set up the environment following the instructions in [INSTALL.md](./docs/INSTALL.md). Then you can run:

```bash
accelerate launch --main_process_port 1234 --config_file simpar/configs/accelerate_configs/zero3.yaml \
    --num_processes=7 simpar/train/llava_trainer_grpo.py \
    --config simpar/configs/config_grpo.yaml \
    --data_path /path_to_annotation_file
```

Note that trl uses 1 separate GPU for online generation (with vLLM), therefore, we recommend you to use at least 2 GPUs for training. Please refer to their documents for more details [here](https://huggingface.co/docs/trl/main/en/grpo_trainer).

We spent lots of time to tune the hyper-parameters and improve the training efficiency. After this, we observed quite promising reward curves 😄:
<br>

<a style="display: block; text-align: left; margin-top: 10px;"><img src="../assets/reward.png" width="60%"></a>