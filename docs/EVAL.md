# Benchmark Evaluation

We provide the scripts to evaluate our model on [GenEval](https://github.com/djghosh13/geneval) and [DPG-Bench](https://github.com/TencentQQGYLab/ELLA/tree/main/dpg_bench) under *./scripts/eval*:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/bench_dpg.sh
```

Please follow the instructions in their repo to calculate the metrics.

# Generate Images with Our Model

We provide a script to generate images with SimpleAR:

```
python3 generate.py --prompts "Your prompt"
```

## serving with vLLM [Optional]

If you want to use vllm to accelerate image generation, please install it from [this repo](https://github.com/wdrink/vllm), we implement classifier-guidance free (CFG) since it is quite important for visual generation:

```
git clone https://github.com/wdrink/vllm

cd vllm

pip install -e .

cd ..

mv vllm vllm_local

mv vllm_local/vllm ./

# reinstall transformer here

pip install "transformers@git+https://github.com/huggingface/transformers.git@7bbc62474391aff64f63fcc064c975752d1fa4de"

```

then just pass **--vllm_serving** to *generate.py* to try vLLM.

## sampling with SJD

We also implement [speculative jacobi decoding (SJD)](https://arxiv.org/abs/2410.01699), you can try it with *--sjd_sampling*.