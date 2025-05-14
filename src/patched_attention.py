from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

class PatchedJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, mode: str, cache_mode: str, save_last_half=True):
        if mode != 'patching' and mode != 'caching' and mode != 'idle':
            raise ValueError('Patched Attention mode must be either patching or caching')
        self.mode = mode
        if cache_mode != 'text_kv' and cache_mode != 'all_v' and cache_mode != "lv":
            raise ValueError('unsupported cache mode')
        self.cache_mode = cache_mode

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("JointAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.save_last_half = save_last_half
        
    def to_patching_mode(self):
        #if not hasattr(self, "cached_key") or not hasattr(self, "cached_value"):
        #    raise ValueError('Key and Value were not cached!')
        self.mode = 'patching'

    def to_caching_mode(self):
        self.mode = 'caching'
        if self.cache_mode == 'text_kv':
            self.cached_text_key = None
            self.cached_text_value = None
        if self.cache_mode == 'all_v':
            self.cached_v = None
            self.cached_text_v = None
        if self.cache_mode == 'lv':
            self.cached_v = None

    def to_idle_mode(self):
        self.mode = 'idle'
        if self.cache_mode == 'text_kv':
            self.cached_text_key = None
            self.cached_text_value = None
        if self.cache_mode == 'all_v':
            self.cached_v = None
            self.cached_text_v = None
        if self.cache_mode == 'lv':
            self.cached_v = None

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            
            
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            
            
            if self.mode == 'patching':
                if self.cache_mode == 'text_kv':
                    encoder_hidden_states_key_proj = self.cached_key
                    encoder_hidden_states_value_proj = self.cached_value
                    print('Patched ', encoder_hidden_states_key_proj.shape)
                if self.cache_mode == 'all_v':
                    value = self.cached_v
                    encoder_hidden_states_value_proj = self.cached_text_v
                if self.cache_mode == 'lv':
                    value = self.cached_v
            elif self.mode == 'caching':
                if self.cache_mode == 'text_kv':
                    if self.save_last_half:
                        self.cached_key = encoder_hidden_states_key_proj.chunk(2)[-1]
                        self.cached_value = encoder_hidden_states_value_proj.chunk(2)[-1]
                    else:
                        self.cached_key = encoder_hidden_states_key_proj
                        self.cached_value = encoder_hidden_states_value_proj
                elif self.cache_mode == 'all_v':
                    self.cached_v = value
                    self.cached_text_v = encoder_hidden_states_value_proj
                elif self.cache_mode == 'lv':
                    self.cached_v = value
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
        
        
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states







