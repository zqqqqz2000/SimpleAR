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