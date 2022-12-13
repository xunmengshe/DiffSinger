from utils.hparams import hparams
from utils.pitch_utils import f0_to_coarse, denorm_f0, norm_f0
import torch
import torch.nn as nn
from torch.nn import functional as F
from modules.commons.common_layers import Embedding, Linear
from modules.fastspeech.tts_modules import FastspeechEncoder, mel2ph_to_dur

class Encoder(FastspeechEncoder):
    def forward_embedding(self, txt_tokens, dur_embed):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        x = x + dur_embed
        if hparams['use_pos_embed']:
            if hparams.get('rel_pos') is not None and hparams['rel_pos']:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(txt_tokens)
                x = x + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, txt_tokens, dur_embed):
        """
        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [T x B x C]
        }
        """
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).detach()
        x = self.forward_embedding(txt_tokens, dur_embed)  # [B, T, H]
        x = super(FastspeechEncoder, self).forward(x, encoder_padding_mask)
        return x

class ParameterEncoder(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.txt_embed = Embedding(len(dictionary), hparams['hidden_size'], dictionary.pad())
        self.dur_embed = Linear(1, hparams['hidden_size'])
        self.encoder = Encoder(self.txt_embed, hparams['hidden_size'], hparams['enc_layers'], hparams['enc_ffn_kernel_size'], num_heads=hparams['num_heads'])
        self.pitch_embed = Embedding(300, hparams['hidden_size'], dictionary.pad())
    
    def forward(self, txt_tokens, mel2ph=None, spk_embed_id=None,
                ref_mels=None, f0=None, uv=None, energy=None, skip_decoder=False,
                spk_embed_dur_id=None, spk_embed_f0_id=None, infer=False, is_slur=None, **kwarg):
        B, T = txt_tokens.shape
        dur = mel2ph_to_dur(mel2ph, T).float()
        dur_embed = self.dur_embed(dur[:, :, None])
        encoder_out = self.encoder(txt_tokens, dur_embed)
        
        decoder_inp = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        decoder_inp = torch.gather(decoder_inp, 1, mel2ph_)
        
        nframes = mel2ph.size(1)
        delta_l = nframes - f0.size(1)
        if delta_l > 0:
            f0 = torch.cat((f0,torch.FloatTensor([[x[-1]] * delta_l for x in f0]).to(f0.device)),1)
        f0 = f0[:,:nframes]
        
        pitch_padding = (mel2ph == 0)
        f0_denorm = denorm_f0(f0, uv, hparams, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)
        pitch_embed = self.pitch_embed(pitch)
        
        ret = {'decoder_inp': decoder_inp + pitch_embed, 'f0_denorm': f0_denorm}
        return ret
