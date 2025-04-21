# SimpleAR: Pushing the Frontier of Autoregressive Visual Generation

[![SimpleAR](https://img.shields.io/badge/Arxiv-SimpleAR-d32f2f.svg?logo=arXiv)](https://arxiv.org/abs/2504.11455)&#160;
<a href='https://huggingface.co/papers/2504.11455'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20-paper-yellow'></a>
<a href='https://huggingface.co/Daniel0724/SimpleAR'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20-checkpoints-blue'></a>
<br>


<div style="text-align: center; margin-top: 0px;">
  <a href="https://arxiv.org/abs/2504.11455" target="_blank" style="text-decoration: none; color: #007acc;">
    SimpleAR: Pushing the Frontier of Autoregressive Visual Generation through Pretraining, SFT, and RL
  </a><br><br>
  <a href="https://wdrink.github.io/">Junke Wang</a><sup>1</sup>, 
<a href="https://zhitian.xyz/">Zhi Tian</a><sup>2</sup>, Xun Wang<sup>2</sup>, Xinyu Zhang<sup>2</sup>, Weilin Huang<sup>2</sup>, <a href="https://zxwu.azurewebsites.net/">Zuxuan Wu</a><sup>1</sup>, Yu-Gang Jiang<sup>1</sup><br>
  <sup>1</sup>Fudan University, <sup>2</sup>ByteDance Seed
  <br>
</div>

<br>

<a style="display: block; text-align: center; margin-top: 20px;"><img src="assets/teaser.png" width="90%"></a>

## Introduction

This paper presents SimpleAR, a vanilla autoregressive visual generation model that achieves state-of-the-art text-to-image generation performance. First the first time, we demonstrate that:
- 🏆 with only 0.5B parameters, an AR model can generate 1024 resolution images with high fidelity, and achieve competitive results on challenging T2I benchmarks, e.g., 0.59 on GenEval and 79.66 on DPG;
- 🚀 both supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO) training could lead to significant improvements on image aesthectics and prompt alignment;
- ⚡️ when deployed with vLLM, the throughput of our model allows for generating 1024 resolution images in 14 seconds, making high-resolution generation practical for real-world applications. 

We open-sourced all the training and inference code, hoping to show the potential of autoregressive visual generation and encourage more participation in this research field.

## Updates

- [2025/04/20] We update the installation instructions and [model zoo](https://huggingface.co/collections/Daniel0724/simplear-6805053f5b4b9961ac025136): thanks [syjmelody](https://github.com/syjmelody) and [wusize](https://github.com/wusize) for raising issues.
- [2025/04/21] Stronger models with better generation quality, and more functionality, e.g., editing and controllable generation, will be released in this repo, please stay tuned!

## Models & Scripts

### Installation

Please follow the instructions below to install the environment:

```bash
python3 -m venv env

source env/bin/activate

pip install -e ".[train]"
```

Note that by default, vllm is not installed. If you want to use vllm, please install it from [this repo](https://github.com/wdrink/vllm), we implement classifier-guidance free (CFG) since it is quite important for visual generation. 

## Model Zoo

We provide both SFT and RL checkpoints:

| name | GenEval | DPG | HF weights 🤗 |
|:---|:---:|:---:|:---:|
| SimpleAR-0.5B-SFT | 0.53 | 79.34 | [simplear-0.5B-sft](https://huggingface.co/Daniel0724/SimpleAR-0.5B-SFT) |
| SimpleAR-0.5B-RL | 0.59 | 79.66 | [simplear-0.5B-grpo](https://huggingface.co/Daniel0724/SimpleAR-0.5B-RL) |
| SimpleAR-1.5B-SFT | 0.61 | 80.11 | [simplear-1.5B-sft](https://huggingface.co/Daniel0724/SimpleAR-1.5B-SFT) |
| SimpleAR-1.5B-RL | 0.63 | 81.31 | [simplear-1.5B-grpo](https://huggingface.co/Daniel0724/SimpleAR-1.5B-RL) |

We use [Cosmos](https://huggingface.co/nvidia/Cosmos-1.0-Tokenizer-DV8x16x16) as our visual tokenizer, you can download and put it under *./checkpoints/*:

```bash
cd checkpoints

git lfs install

git clone https://huggingface.co/nvidia/Cosmos-1.0-Tokenizer-DV8x16x16
```

## Play with Our Model

You can directly load SimpleAR with *from_pretrained* now 🤗 !

```python
import os
import torch
from torchvision.utils import save_image
from transformers import AutoTokenizer
from simpar.model.tokenizer.cosmos_tokenizer.networks import TokenizerConfigs
from simpar.model.tokenizer.cosmos_tokenizer.video_lib import CausalVideoTokenizer as CosmosTokenizer
from simpar.model.language_model.simpar_qwen2 import SimpARForCausalLM

device = "cuda:0"
model_name = "Daniel0724/SimpleAR-0.5B-RL"

# define your prompt here:
prompt = "Inside a warm room with a large window showcasing a picturesque winter landscape, three gleaming ruby red necklaces are elegantly laid out on the plush surface of a deep purple velvet jewelry box. The gentle glow from the overhead light accentuates the rich color and intricate design of the necklaces. Just beyond the glass pane, snowflakes can be seen gently falling to coat the ground outside in a blanket of white."

# Load LLM and tokenizer
model = SimpARForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load Cosmos tokenizer
tokenizer_config = TokenizerConfigs["DV"].value
tokenizer_config.update(dict(spatial_compression=16, temporal_compression=8))
vq_model = CosmosTokenizer(checkpoint_enc=f"./checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/encoder.jit", checkpoint_dec=f"./checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16/decoder.jit", tokenizer_config=tokenizer_config)
vq_model.eval()
vq_model.requires_grad_(False)
codebook_size = 64000
latent_size = 64

format_prompt = "<|t2i|>" + "A highly realistic image of " + prompt + "<|soi|>"
input_ids = tokenizer(format_prompt, return_tensors="pt").input_ids.to(device)
uncond_prompt = "<|t2i|>" + "An image of aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion" + "<|soi|>"
uncond_input_ids = tokenizer(uncond_prompt, return_tensors="pt").input_ids.to(device)

# next token prediction
with torch.inference_mode():
    output_ids = model.generate_visual(
        input_ids,
        negative_prompt_ids=uncond_input_ids,
        cfg_scale=6.0,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=64000,
        max_new_tokens=4096,
        use_cache=True
    )

index_sample = output_ids[:, input_ids.shape[1]: input_ids.shape[1] + 4096].clone()
index_sample = index_sample - len(tokenizer)
index_sample = torch.clamp(index_sample, min=0, max=codebook_size-1)
index_sample = index_sample.reshape(-1, latent_size, latent_size).unsqueeze(1)

# decode with tokenizer
with torch.inference_mode():
    samples = vq_model.decode(index_sample)

samples = samples.squeeze(2)
save_image(samples, os.path.join(f"{prompt[:50]}.png"), normalize=True, value_range=(-1, 1))
```

## Training

### Data Preparation

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

### Pretrain and SFT

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

### GRPO

We follow [Open-R1](https://github.com/huggingface/open-r1) to implement GRPO training with [trl](https://github.com/huggingface/trl), please refer to *./scripts/env_rl.sh* to set up the environment. Then you can run:

```bash
accelerate launch --main_process_port 1234 --config_file simpar/configs/accelerate_configs/zero3.yaml \
    --num_processes=7 simpar/train/llava_trainer_grpo.py \
    --config simpar/configs/config_grpo.yaml \
    --data_path /path_to_annotation_file
```

We spent lots of time to tune the hyper-parameters and improve the training efficiency. After this, we observed quite promising reward curves 😄:
<br>

<a style="display: block; text-align: left; margin-top: 10px;"><img src="assets/reward.png" width="60%"></a>


## Evaluation and Inference

### Benchmark Evaluation 

We provide the scripts to evaluate our model on [GenEval](https://github.com/djghosh13/geneval) and [DPG-Bench](https://github.com/TencentQQGYLab/ELLA/tree/main/dpg_bench) under *./scripts/eval*:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/bench_dpg.sh
```

Please follow the instructions in their repo to calculate the metrics.

### Inference Acceleration

We provide a script to generate images with our model:

```
python3 generate.py --prompts "Your prompt"
```


#### serving with vLLM
vLLM could significantly improve the inference efficiency, you can first install it from [this repo](https://github.com/wdrink/vllm):

```
git clone https://github.com/wdrink/vllm

cd vllm

pip install -e .

cd ..

mv vllm vllm_local

mv vllm_local/vllm ./
```

then pass **--vllm_serving** to *generate.py* to try vLLM.

#### sampling with SJD

We also implement [speculative jacobi decoding (SJD)](https://arxiv.org/abs/2410.01699), you can try it with *--sjd_sampling*.


## Visualizations

<p align="left">
  <img src="./assets/visualization.png" alt="" width="80%" />
  <img src="./assets/geneval.png" alt="" width="80%" />
  <br>
  <em>1024 x 1024 generation results by SimpleAR.</em>
</p>

## Citation
If you find this repository helpful, please consider citing:
```bib
@article{wang2025simplear,
  title={SimpleAR: Pushing the Frontier of Autoregressive Visual Generation through Pretraining, SFT, and RL},
  author={Wang, Junke and Tian, Zhi and Wang, Xun and Zhang, Xinyu and Huang, Weilin and Wu, Zuxuan and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2504.11455},
  year={2025}
}
```

## Acknowledgement

We thank [Peize Sun](https://peizesun.github.io/), [Rui Tian](https://scholar.google.com/citations?user=zTI-OFoAAAAJ&hl=en), [Feng Li](https://fengli-ust.github.io/), and [Teng Yao](https://tyshiwo1.github.io/) for their valuable discussions.
