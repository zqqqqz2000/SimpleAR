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

- [2025/04/20] [Installation instructions](./docs/TRAIN.md) and [model zoo](https://huggingface.co/collections/Daniel0724/simplear-6805053f5b4b9961ac025136) are updated! Thanks [syjmelody](https://github.com/syjmelody), [wusize](https://github.com/wusize), and [micky-li-hd](https://github.com/micky-li-hd) for raising issues.
- [2025/04/21] Stronger models with better generation quality, and more functionality, e.g., editing and controllable generation, will be released in this repo, please stay tuned!
- [2025/04/22] We provide a [demo code](./docs/PLAY.md) to play with our released models.

## Installation

For basic usage (pretraining, SFT, inference without vLLM), you can install the dependencies with:

```bash
python3 -m venv env

source env/bin/activate

pip install -e ".[train]"
```

While for advanced usage, please refer to [TRAIN.md](./docs/TRAIN.md) (GRPO training) and [EVAL.md](./docs/EVAL.md) (inference with vLLM) to setup the environments, respectively.

## Models & Scripts

## Model Zoo

We provide both SFT and RL checkpoints:

| name | GenEval | DPG | HF weights 🤗 |
|:---|:---:|:---:|:---:|
| SimpleAR-0.5B-SFT | 0.53 | 79.34 | [simplear-0.5B-sft](https://huggingface.co/Daniel0724/SimpleAR-0.5B-SFT) |
| SimpleAR-0.5B-RL | 0.59 | 79.66 | [simplear-0.5B-grpo](https://huggingface.co/Daniel0724/SimpleAR-0.5B-RL) |
| SimpleAR-1.5B-SFT | 0.61 | 80.11 | [simplear-1.5B-sft](https://huggingface.co/Daniel0724/SimpleAR-1.5B-SFT) |
| SimpleAR-1.5B-RL | 0.63 | 81.31 | [simplear-1.5B-grpo](https://huggingface.co/Daniel0724/SimpleAR-1.5B-RL) |

[Cosmos](https://huggingface.co/nvidia/Cosmos-1.0-Tokenizer-DV8x16x16) is used as our visual tokenizer, you can download and put it under *./checkpoints/*:

```bash
cd checkpoints

git lfs install

git clone https://huggingface.co/nvidia/Cosmos-1.0-Tokenizer-DV8x16x16
```

## Play with Our Model (Quick Start)

You can directly load SimpleAR with *from_pretrained* now 🤗! We provide the demo code in [PLAY.md](./docs/PLAY.md).

## Training

Please find the instructions on data preparation and training [here](./docs/TRAIN.md).

## Evaluation and Inference

We provide scripts to evaluate our released checkpoints on [GenEval](https://github.com/djghosh13/geneval) and [DPG-Bench](https://github.com/TencentQQGYLab/ELLA/tree/main/dpg_bench).
Please see [EVAL.md](./docs/EVAL.md) for more details.

Also, you can generate images with SimpleAR using *generate.py*. We implement different acceleration approaches, e.g., vLLM, [speculative jacobi decoding](https://arxiv.org/abs/2410.01699). Please refer to [EVAL.md](./docs/EVAL.md).

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
