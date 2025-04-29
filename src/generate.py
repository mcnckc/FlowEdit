import torch
from diffusers import StableDiffusion3Pipeline
from omegaconf import OmegaConf
from diffusers.hooks import apply_group_offloading
from patched_attention2 import PatchedJointAttnProcessor2_0

def load_config():
    conf_cli = OmegaConf.from_cli()
    config_path = conf_cli.config_path
    conf_file = OmegaConf.load(config_path)
    config = OmegaConf.merge(conf_file, conf_cli)
    return config

if __name__ == "__main__":   
    #cfg = load_config()
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    if True:
        print("OFFLOADING")
        pipe.text_encoder = pipe.text_encoder.to(device)
        pipe.text_encoder_2 = pipe.text_encoder_2.to(device)
        pipe.vae = pipe.vae.to(device)
        apply_group_offloading(pipe.text_encoder_3, onload_device=device, offload_device=torch.device('cpu'), 
                                            offload_type="block_level", num_blocks_per_group=12, use_stream=True)
        apply_group_offloading(pipe.transformer, onload_device=device, offload_device=torch.device('cpu'), 
                                            offload_type="block_level", num_blocks_per_group=11, use_stream=True)
    else:
        print("NO OFFLOAD")
        pipe = pipe.to(device)
    #src_prompt = "KEEP CALM AND CARRY ON, image contains text that reads \"KEEP CALM AND CARRY ON\""
    #tar_prompt = "KEEP Salt AND CARRY ON, image contains text that reads \"KEEP Salt AND CARRY ON\""
    src_prompt = "A cute dog, holding a sign with text \"FOOD\""
    tar_prompt = "A cute dog, holding a sign with text \"HELLO\""
    latents = pipe.prepare_latents(
        1,
        pipe.transformer.config.in_channels,
        512,
        512,
        pipe.dtype,
        device,
        generator=None
    )
    src_im = pipe(
        prompt=src_prompt,
        negative_prompt="",
        num_inference_steps=50,
        height=512,
        width=512,
        guidance_scale=7.0,
        latents=latents
    ).images[0]

    pipe.transformer.transformer_blocks[10].attn.set_processor(PatchedJointAttnProcessor2_0(mode='caching'))
    pipe.transformer.transformer_blocks[10].attn.processor.to_caching_mode()
    _ = pipe(
        prompt=tar_prompt,
        negative_prompt="",
        num_inference_steps=50,
        height=512,
        width=512,
        guidance_scale=7.0,
        latents=latents
    ).images[0]
    pipe.transformer.transformer_blocks[10].attn.processor.to_patching_mode()
    tar_im = pipe(
        prompt=src_prompt,
        negative_prompt="",
        num_inference_steps=50,
        height=512,
        width=512,
        guidance_scale=7.0,
        latents=latents
    ).images[0]
    src_im.save("src.png")
    tar_im.save("tar.png")