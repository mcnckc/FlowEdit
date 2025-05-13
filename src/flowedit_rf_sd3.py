from typing import Optional, Tuple, Union
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
import numpy as np
from .patched_attention import PatchedJointAttnProcessor2_0
from .flowedit_utils import scale_noise
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


def calc_v_sd3(pipe, latent_model_input, prompt_embeds, pooled_prompt_embeds, guidance_scale, t):
    timestep = t.expand(latent_model_input.shape[0])
    with torch.no_grad():
        # # predict the noise for the source prompt
        noise_pred_src_tar = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # perform guidance source
        #if pipe.do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred_src_tar.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred

def calc_v_sd3_patched(pipe, tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, tar_guidance_scale, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(tar_latent_model_input.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i

    with torch.no_grad():
        pipe.transformer.transformer_blocks[10].attn.processor.to_caching_mode()
        # # predict the noise for the source prompt
        _ = pipe.transformer(
            hidden_states=tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds.chunk(2)[1],
            pooled_projections=src_tar_pooled_prompt_embeds.chunk(2)[1],
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        pipe.transformer.transformer_blocks[10].attn.processor.to_patching_mode()
        noise_pred_tar = pipe.transformer(
            hidden_states=tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds.chunk(2)[0],
            pooled_projections=src_tar_pooled_prompt_embeds.chunk(2)[0],
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        #if pipe.do_classifier_free_guidance:
        tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_tar.chunk(2)
        noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)
        pipe.transformer.transformer_blocks[10].attn.processor.to_caching_mode()

    return noise_pred_tar

def rf_v_sd3(z, pipe, prompt_embeds, pooled_prompt_embeds, guidance_scale, rt, lt):
    dt = (lt - rt) / 2
    v = calc_v_sd3(pipe, torch.cat([z, z]), prompt_embeds, pooled_prompt_embeds, guidance_scale, rt * 1000)
    zmid = z + v * dt
    vmid = calc_v_sd3(pipe, torch.cat([zmid, zmid]), prompt_embeds, pooled_prompt_embeds, guidance_scale, (rt + dt) * 1000)
    dv = (vmid - v) / dt
    return (lt - rt) * v + 1 / 2 * ((lt - rt) ** 2) * dv

def patched_rf_v_sd3(z, pipe, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, guidance_scale, rt, lt):
    dt = (lt - rt) / 2
    v = calc_v_sd3_patched(pipe, torch.cat([z, z]), src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, guidance_scale, rt * 1000)
    zmid = z + v * dt
    vmid = calc_v_sd3_patched(pipe, torch.cat([zmid, zmid]), src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, guidance_scale, (rt + dt) * 1000)
    dv = (vmid - v) / dt
    return (lt - rt) * v + 1 / 2 * ((lt - rt) ** 2) * dv

@torch.no_grad()
def FlowEditRFSD3(pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    scene_text_edit=True):
    
    device = x_src.device

    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)

    num_warmup_steps = max(len(timesteps) - T_steps * scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale
    print("EDIT1")
    print("GUIDE", pipe.do_classifier_free_guidance)
    # src prompts
    (
        src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        device=device,
    )
    print(src_prompt_embeds,
        src_negative_prompt_embeds,
        src_pooled_prompt_embeds,
        src_negative_pooled_prompt_embeds)
    print("EDIT2")
    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_negative_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        device=device,
    )
    print("EDIT3")
    # CFG prep
    src_tar_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds, tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src.clone()
    if scene_text_edit:
        pipe.transformer.transformer_blocks[10].attn.set_processor(PatchedJointAttnProcessor2_0(mode='caching', save_last_half=False))
    print("EDIT LOOP")
    print(len(timesteps))
    print(timesteps)
    for i, t in tqdm(enumerate(timesteps)):
        
        if T_steps - i > n_max:
            continue
        
        t_i = t/1000
        if i+1 < len(timesteps): 
            t_im1 = (timesteps[i+1])/1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)
        
        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                
                zt_src = (1-t_i)*x_src + (t_i)*fwd_noise

                zt_tar = zt_edit + zt_src - x_src

                src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar]) if True else (zt_src, zt_tar)
                v_src = rf_v_sd3(zt_src, pipe, src_tar_prompt_embeds.chunk(2)[0], src_tar_pooled_prompt_embeds.chunk(2)[0], src_guidance_scale, t_i, t_im1)
                if scene_text_edit:
                    v_tar = patched_rf_v_sd3(zt_tar, pipe, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, tar_guidance_scale, t_i, t_im1)
                else:
                    v_tar = rf_v_sd3(zt_tar, pipe, src_tar_prompt_embeds.chunk(2)[1], src_tar_pooled_prompt_embeds.chunk(2)[1], tar_guidance_scale, t_i, t_im1)
                #print("DIFF:", (v_tar - v_src).abs().max(), (v_tar - v_src).abs().mean())
                V_delta_avg += (1/n_avg) * (v_tar - v_src)

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + V_delta_avg
            #print("ZDIFF:", (zt_edit - x_src).abs().max(), (zt_edit - x_src).abs().mean())
            #print("RZDIFF:", ((zt_edit - x_src) / (x_src + 1e-7)).abs().max(), ((zt_edit - x_src) / (x_src + 1e-7)).abs().mean())
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else: # i >= T_steps-n_min # regular sampling for last n_min steps
            print("NMIN CASE:")
            if i == T_steps-n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src
                
            src_tar_latent_model_input = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar]) if pipe.do_classifier_free_guidance else (xt_src, xt_tar)

            _, Vt_tar = calc_v_sd3_patched(pipe, src_tar_latent_model_input,src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(xt_tar.dtype)

            xt_tar = prev_sample
        
    return zt_edit if n_min == 0 else xt_tar