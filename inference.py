import torch
from omegaconf import OmegaConf
from diffusers import StableDiffusion3Pipeline
from diffusers import FluxPipeline
from diffusers.hooks import apply_group_offloading
from PIL import Image
import random 
import numpy as np
import yaml
import os
from src.flowedit_sd3 import FlowEditSD3, FlowEditSD3Embeds, get_text_embeds
from src.flowedit_rf_sd3 import FlowEditRFSD3
from src.flowedit_flux import FlowEditFLUX
from tqdm import tqdm

def load_config():
    conf_cli = OmegaConf.from_cli()
    config_path = conf_cli.config_path
    conf_file = OmegaConf.load(config_path)
    config = OmegaConf.merge(conf_file, conf_cli)
    return config

if __name__ == "__main__":
    cfg = load_config()
    # set device
    device_number = cfg.device_number
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

    # load exp yaml file to dict

    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")
    model_type = cfg.model_type # currently only one model type per run
    print('Using device:', device)

    if model_type == 'FLUX':
        # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16) 
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
        pipe = pipe.to(device)
    elif model_type == 'SD3':
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        if cfg.offload_half_layers:
            print("OFFLOADING")
            pipe.text_encoder = pipe.text_encoder.to(device)
            #pipe.text_encoder_2 = pipe.text_encoder_2.to(device)
            pipe.vae = pipe.vae.to(device)
            apply_group_offloading(pipe.text_encoder_2, onload_device=device, offload_device=torch.device('cpu'), 
                                            offload_type="block_level", num_blocks_per_group=2, use_stream=True)
            apply_group_offloading(pipe.text_encoder_3, onload_device=device, offload_device=torch.device('cpu'), 
                                            offload_type="block_level", num_blocks_per_group=2, use_stream=True)
            apply_group_offloading(pipe.transformer, onload_device=device, offload_device=torch.device('cpu'), 
                                            offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        else:
            print("NO OFFLOAD")
            pipe = pipe.to(device)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    
    
    scheduler = pipe.scheduler
    print("LOADED TO GPU")
    for exp_id, exp_dict in enumerate(cfg.exps):
        for tar_guidance_scale in exp_dict["tar_guidance_scale"]:
            if exp_dict['add_eq']:
                src_scales = list(set(exp_dict["src_guidance_scale"] + [tar_guidance_scale]))
            else:
                src_scales = exp_dict["src_guidance_scale"]
            for src_guidance_scale in src_scales:
                if src_guidance_scale > tar_guidance_scale:
                    continue
                for patch_v_layers in exp_dict['patch_v_layers']:
                    for patch_v_steps in exp_dict['patch_v_steps']:
                        exp_name = exp_dict["exp_name"]
                        # model_type = exp_dict["model_type"]
                        T_steps = exp_dict["T_steps"]
                        n_avg = exp_dict["n_avg"]
                        n_min = exp_dict["n_min"]
                        seed = exp_dict["seed"]
                        scene_text_edit = exp_dict.scene_text_edit
                        # set seed
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        dataset_yaml = exp_dict["dataset_yaml"]
                        with open(dataset_yaml) as file:
                            dataset_configs = yaml.load(file, Loader=yaml.FullLoader)

                        # check dataset_configs 
                        for data_dict in dataset_configs:
                            tar_prompts = data_dict["target_prompts"]

                        for data_dict in tqdm(dataset_configs[:cfg.max_edits] if 'max_edits' in cfg else dataset_configs):
                            #for data_dict in tqdm(dataset_configs[7:8]):
                            src_prompt = data_dict["source_prompt"]
                            tar_prompts = data_dict["target_prompts"]
                            print("SRC:", src_prompt)
                            print("TGT:", tar_prompts)
                            negative_prompt =  "" # optionally add support for negative prompts (SD3)
                            image_src_path = data_dict["input_img"]

                                # load image
                            image = Image.open(image_src_path)
                                # crop image to have both dimensions divisibe by 16 - avoids issues with resizing
                            image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
                            image_src = pipe.image_processor.preprocess(image)
                                # cast image to half precision
                            image_src = image_src.to(device).half()
                            with torch.autocast("cuda"), torch.inference_mode():
                                x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
                            x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                            # send to cuda
                            x0_src = x0_src.to(device)
                                
                            for tar_num, tar_prompt in enumerate(tar_prompts):
                                for n_max in exp_dict["n_max"]:
                                    print("START EDIT")
                                    if model_type == 'SD3':
                                        if cfg.use_rf:
                                            x0_tar = FlowEditRFSD3(pipe,
                                                                                    scheduler,
                                                                                    x0_src,
                                                                                    src_prompt,
                                                                                    tar_prompt,
                                                                                    negative_prompt,
                                                                                    T_steps,
                                                                                    n_avg,
                                                                                    src_guidance_scale,
                                                                                    tar_guidance_scale,
                                                                                    patch_v_layers,
                                                                                    patch_v_steps,
                                                                                    cfg.v_cache_mode,
                                                                                    n_min,
                                                                                    n_max,
                                                                                    cfg.dtc,
                                                                                    scene_text_edit=scene_text_edit)
                                        else:
                                            #text_embs, text_pooled_embs = get_text_embeds(pipe, scheduler, x0_src, src_prompt, tar_prompt, negative_prompt, T_steps, src_guidance_scale, tar_guidance_scale)
                                            x0_tar = FlowEditSD3(pipe,
                                                                                    scheduler,
                                                                                    x0_src,
                                                                                    src_prompt,
                                                                                    tar_prompt,
                                                                                    negative_prompt,
                                                                                    T_steps,
                                                                                    n_avg,
                                                                                    src_guidance_scale,
                                                                                    tar_guidance_scale,
                                                                                    n_min,
                                                                                    n_max,
                                                                                    scene_text_edit=scene_text_edit)
                                        
                                        
                                    elif model_type == 'FLUX':
                                        x0_tar = FlowEditFLUX(pipe,
                                                                                scheduler,
                                                                                x0_src,
                                                                                src_prompt,
                                                                                tar_prompt,
                                                                                negative_prompt,
                                                                                T_steps,
                                                                                n_avg,
                                                                                src_guidance_scale,
                                                                                tar_guidance_scale,
                                                                                n_min,
                                                                                n_max,)
                                    else:
                                        raise NotImplementedError(f"Sampler type {model_type} not implemented")

                                    """
                                    print("DONE edit")
                                    print("ABS:", (x0_src - x0_tar).abs().mean(), (x0_src - x0_tar).abs().max())
                                    print("ABS:", (x0_src - x0_tar2).abs().mean(), (x0_src - x0_tar2).abs().max())
                                    print("RABS:", ((x0_src - x0_tar) / x0_src).abs().mean(), ((x0_src - x0_tar) / x0_src).abs().max())
                                    print("RABS:", ((x0_src - x0_tar2) / x0_src).abs().mean(), ((x0_src - x0_tar2) / x0_src).abs().max())
                                    """
                                    x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                                    with torch.autocast("cuda"), torch.inference_mode():
                                        image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
                                    image_tar = pipe.image_processor.postprocess(image_tar)

                            
                                    src_prompt_txt = data_dict["input_img"].split("/")[-1].split(".")[0]

                                    tar_prompt_txt = str(tar_num)
                                    print("DONE postprocess")
                                    # make sure to create the directories before saving
                                    save_dir = f"outputs/{exp_name}/{model_type}/src_{src_prompt_txt}/tar_{tar_prompt_txt}"
                                    os.makedirs(save_dir, exist_ok=True)
                                    images_dir = "results/" + exp_name
                                    os.makedirs(images_dir, exist_ok=True)
                                    fname = os.path.basename(image_src_path).split('.')
                                    image_tar[0].save(f"{images_dir}/{fname[0]}-{tar_num}-{src_guidance_scale}-{tar_guidance_scale}-{n_max}.{fname[1]}")
                                    #image_tar[0].save(f"{save_dir}/output_T_steps_{T_steps}_n_avg_{n_avg}_cfg_enc_{src_guidance_scale}_cfg_dec{tar_guidance_scale}_n_min_{n_min}_n_max_{n_max}_seed{seed}.png")
                                    # also save source and target prompt in txt file
                                    with open(f"{save_dir}/prompts.txt", "w") as f:
                                        f.write(f"Source prompt: {src_prompt}\n")
                                        f.write(f"Target prompt: {tar_prompt}\n")
                                        f.write(f"Seed: {seed}\n")
                                        f.write(f"Sampler type: {model_type}\n")
                                    print("Saved")
                

    print("Done")