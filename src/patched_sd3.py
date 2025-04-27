import torch

def calc_v_sd3_patched(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t):
    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timestep = t.expand(src_tar_latent_model_input.shape[0])
    # joint_attention_kwargs = {}
    # # add timestep to joint_attention_kwargs
    # joint_attention_kwargs["timestep"] = timestep[0]
    # joint_attention_kwargs["timestep_idx"] = i

    with torch.no_grad():
        pipe.transformer.transformer_blocks[10].attn.processor.to_caching_mode()
        # # predict the noise for the source prompt
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # perform guidance source
        if pipe.do_classifier_free_guidance:
            src_noise_pred_uncond, src_noise_pred_text, _, _ = noise_pred_src_tar.chunk(4)
            noise_pred_src = src_noise_pred_uncond + src_guidance_scale * (src_noise_pred_text - src_noise_pred_uncond)

        pipe.transformer.transformer_blocks[10].attn.processor.to_patching_mode()
        noise_pred_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input.chunk(2)[0],
            timestep=timestep.chunk(2)[0],
            encoder_hidden_states=src_tar_prompt_embeds.chunk(2)[0],
            pooled_projections=src_tar_pooled_prompt_embeds.chunk(2)[0],
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if pipe.do_classifier_free_guidance:
            tar_noise_pred_uncond, tar_noise_pred_text = noise_pred_tar.chunk(2)
            noise_pred_tar = tar_noise_pred_uncond + tar_guidance_scale * (tar_noise_pred_text - tar_noise_pred_uncond)

    return noise_pred_src, noise_pred_tar