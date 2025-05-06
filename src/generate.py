import torch
from diffusers import StableDiffusion3Pipeline
from omegaconf import OmegaConf
from diffusers.hooks import apply_group_offloading
from diffusers.models.attention_processor import JointAttnProcessor2_0
from patched_attention2 import PatchedJointAttnProcessor2_0
import os

def load_config():
    conf_cli = OmegaConf.from_cli()
    conf_file = OmegaConf.load(conf_cli.default_config)
    config = OmegaConf.merge(conf_file, conf_cli)
    return config

if __name__ == "__main__":   
    cfg = load_config()
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    if cfg.offload:
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

    src_prompt = "Corgi dog with a sign saying \"food\""
    tar_prompt = "Corgi dog with a sign saying \"hello\""
    #src_prompt = "Road sign with text \"Singapore\" on it"
    #tar_prompt = "Road sign with text \"Localize\" on it"
    """
    os.makedirs('corgi-results', exist_ok=True)
    for i in range(20):
        im = pipe(
            prompt=src_prompt,
            negative_prompt="",
            num_inference_steps=50,
            height=cfg.imsize,
            width=cfg.imsize,
            guidance_scale=8.5
        ).images[0]
        im.save('corgi-results/' + str(i) + '.png')
    """
    #src_prompt = "KEEP CALM AND CARRY ON, image contains text that reads \"KEEP CALM AND CARRY ON\""
    #tar_prompt = "KEEP Salt AND CARRY ON, image contains text that reads \"KEEP Salt AND CARRY ON\""
    #cfgs = [5, 6, 7, 8, 9, 10, 11]
    cfgs = [7, 7.5, 8, 8.5, 9, 10]
    nmaxs = list(range(40, 51))
    layers = [8, 9, 10, 11, 12]
    layers_confs = [layers, [10]]
    for smp in range(2):
        latents = pipe.prepare_latents(
                1,
                pipe.transformer.config.in_channels,
                cfg.imsize,
                cfg.imsize,
                pipe.dtype,
                device,
                generator=None
        )
        for layers in layers_confs:
            cur_path = 'corgi-results/' + str(smp)
            os.makedirs(cur_path, exist_ok=True)
            for cur_cfg in cfgs:
                os.makedirs(cur_path, exist_ok=True)
                for layer in layers:
                    pipe.transformer.transformer_blocks[layer].attn.set_processor(JointAttnProcessor2_0())
                src_im = pipe(
                    prompt=src_prompt,
                    negative_prompt="",
                    num_inference_steps=50,
                    height=cfg.imsize,
                    width=cfg.imsize,
                    guidance_scale=cur_cfg,
                    latents=latents
                ).images[0]
                
                for layer in layers:
                    pipe.transformer.transformer_blocks[layer].attn.set_processor(PatchedJointAttnProcessor2_0(mode='caching', patching_step=50))
                    pipe.transformer.transformer_blocks[layer].attn.processor.to_caching_mode()
                mid_im = pipe(
                    prompt=tar_prompt,
                    negative_prompt="",
                    num_inference_steps=50,
                    height=cfg.imsize,
                    width=cfg.imsize,
                    guidance_scale=cur_cfg,
                    latents=latents
                ).images[0]
                for nmax in nmaxs:
                    for layer in layers:
                        pipe.transformer.transformer_blocks[layer].attn.processor.to_patching_mode()
                        pipe.transformer.transformer_blocks[layer].attn.processor.patching_step = nmax
                    tar_im = pipe(
                        prompt=src_prompt,
                        negative_prompt="",
                        num_inference_steps=50,
                        height=cfg.imsize,
                        width=cfg.imsize,
                        guidance_scale=cur_cfg,
                        latents=latents
                    ).images[0]
                    tar_im.save(f"{cur_path}/tar-cfg{cur_cfg}-nmax{nmax}-ls{len(layers)}.png")
                for layer in layers:
                    pipe.transformer.transformer_blocks[layer].attn.set_processor(JointAttnProcessor2_0())
            
                src_im.save(f"{cur_path}/src-cfg{cur_cfg}.png")
                mid_im.save(f"{cur_path}/mid-cfg{cur_cfg}.png")
        
        
            
    