from hcpdiff.models.wrapper import TEUnetWrapper
from hcpdiff.utils import pad_attn_bias
import torch
from .posemb_layers import get_2d_rotary_pos_embed, get_fill_resize_and_crop

class DITWrapper(TEUnetWrapper):
    patch_size = 2
    seq_len = [77, 256]

    def calc_rope(self, height, width):
        th = height  // self.patch_size
        tw = width  // self.patch_size
        base_size = 512 // 8 // self.patch_size
        start, stop = get_fill_resize_and_crop((th, tw), base_size)
        sub_args = [start, stop, (th, tw)]
        rope = get_2d_rotary_pos_embed(self.unet.hidden_size/self.unet.num_heads, *sub_args)
        return rope
    
    def forward(self, prompt_ids, noisy_latents, timesteps, attn_mask=None, position_ids=None, crop_info=None, plugin_input={}, **kwargs):
        input_all = dict(prompt_ids=prompt_ids, noisy_latents=noisy_latents, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask, **plugin_input)

        if hasattr(self.TE, 'input_feeder'):
            for feeder in self.TE.input_feeder:
                feeder(input_all)
        encoder_hidden_states = self.TE(prompt_ids, position_ids=position_ids, attention_mask=attn_mask, output_hidden_states=True)[0]  # Get the text embedding for conditioning

        if attn_mask is not None:
            attn_mask = attn_mask.split(self.seq_len, dim=1)
            encoder_hidden_states_list = []
            attn_mask_list = []
            for attn_mask_i, encoder_hidden_states_i in zip(attn_mask, encoder_hidden_states):
                attn_mask_i[:, :self.min_attnmask] = 1
                #encoder_hidden_states_i, attn_mask_i = pad_attn_bias(encoder_hidden_states_i, attn_mask_i)
                encoder_hidden_states_list.append(encoder_hidden_states_i)
                attn_mask_list.append(attn_mask_i)
            encoder_hidden_states = encoder_hidden_states_list
            attn_mask = attn_mask_list

        input_all['encoder_hidden_states'] = encoder_hidden_states
        if hasattr(self.unet, 'input_feeder'):
            for feeder in self.unet.input_feeder:
                feeder(input_all)
        style = torch.as_tensor([0] * noisy_latents.shape[0], device=noisy_latents.device)
        freqs_cis_img = self.calc_rope(noisy_latents.shape[2],noisy_latents.shape[3])
        
        model_pred = self.unet(x = noisy_latents,
                        t = timesteps,
                        encoder_hidden_states=encoder_hidden_states[0],
                        text_embedding_mask=attn_mask[0],
                        encoder_hidden_states_t5=encoder_hidden_states[1],
                        text_embedding_mask_t5=attn_mask[1],
                        image_meta_size=crop_info,
                        style=style,
                        cos_cis_img=freqs_cis_img[0],
                        sin_cis_img=freqs_cis_img[1],
                        return_dict=False)  # Predict the noise residual
        return model_pred