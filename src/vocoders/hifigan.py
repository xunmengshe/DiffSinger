import glob
import json
import os
import re

import librosa
import torch

import utils
from modules.hifigan.hifigan import HifiGanGenerator
from utils.hparams import hparams, set_hparams
from src.vocoders.base_vocoder import register_vocoder
from src.vocoders.pwg import PWG
from src.vocoders.vocoder_utils import denoise


def load_model(config_path, file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext = os.path.splitext(file_path)[-1]
    if ext == '.pth':
        if '.yaml' in config_path:
            config = set_hparams(config_path, global_hparams=False)
        elif '.json' in config_path:
            config = json.load(open(config_path, 'r', encoding='utf-8'))
        model = torch.load(file_path, map_location="cpu")
    elif ext == '.ckpt':
        ckpt_dict = torch.load(file_path, map_location="cpu")
        if '.yaml' in config_path:
            config = set_hparams(config_path, global_hparams=False)
            state = ckpt_dict["state_dict"]["model_gen"]
        elif '.json' in config_path:
            config = json.load(open(config_path, 'r', encoding='utf-8'))
            state = ckpt_dict["generator"]
        model = HifiGanGenerator(config)
        model.load_state_dict(state, strict=True)
        model.remove_weight_norm()
    model = model.eval().to(device)
    print(f"| Loaded model parameters from {file_path}.")
    print(f"| HifiGAN device: {device}.")
    return model, config, device


total_time = 0


@register_vocoder
class HifiGAN(PWG):
    def __init__(self):
        base_dir = hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        file_path = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.*'), key=
        lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).*', x.replace('\\','/'))[0]))[-1]
        print('| load HifiGAN: ', file_path)
        self.model, self.config, self.device = load_model(config_path=config_path, file_path=file_path)

    def spec2wav_torch(self, mel, **kwargs):
        if self.config['audio_sample_rate'] != hparams['audio_sample_rate']:
            print('Mismatch parameters: hparams[\'audio_sample_rate\']=',hparams['audio_sample_rate'],'!=',self.config['audio_sample_rate'],'(vocoder)')
        if self.config['audio_num_mel_bins'] != hparams['audio_num_mel_bins']:
            print('Mismatch parameters: hparams[\'audio_num_mel_bins\']=',hparams['audio_num_mel_bins'],'!=',self.config['audio_num_mel_bins'],'(vocoder)')
        if self.config['fft_size'] != hparams['fft_size']:
            print('Mismatch parameters: hparams[\'fft_size\']=',hparams['fft_size'],'!=',self.config['fft_size'],'(vocoder)')
        if self.config['win_size'] != hparams['win_size']:
            print('Mismatch parameters: hparams[\'win_size\']=',hparams['win_size'],'!=',self.config['win_size'],'(vocoder)')
        if self.config['hop_size'] != hparams['hop_size']:
            print('Mismatch parameters: hparams[\'hop_size\']=',hparams['hop_size'],'!=',self.config['hop_size'],'(vocoder)')
        if self.config['fmin'] != hparams['fmin']:
            print('Mismatch parameters: hparams[\'fmin\']=',hparams['fmin'],'!=',self.config['fmin'] ,'(vocoder)')
        if self.config['fmax'] != hparams['fmax']:
            print('Mismatch parameters: hparams[\'fmax\']=',hparams['fmax'],'!=',self.config['fmax'] ,'(vocoder)')
        with torch.no_grad():
            c = mel.transpose(2, 1)
            f0 = kwargs.get('f0')
            if f0 is not None and hparams.get('use_nsf'):
                y = self.model(c, f0).view(-1)
            else:
                y = self.model(c).view(-1)
            return y

    def spec2wav(self, mel, **kwargs):
        if self.config['audio_sample_rate'] != hparams['audio_sample_rate']:
            print('Mismatch parameters: hparams[\'audio_sample_rate\']=',hparams['audio_sample_rate'],'!=',self.config['audio_sample_rate'],'(vocoder)')
        if self.config['audio_num_mel_bins'] != hparams['audio_num_mel_bins']:
            print('Mismatch parameters: hparams[\'audio_num_mel_bins\']=',hparams['audio_num_mel_bins'],'!=',self.config['audio_num_mel_bins'],'(vocoder)')
        if self.config['fft_size'] != hparams['fft_size']:
            print('Mismatch parameters: hparams[\'fft_size\']=',hparams['fft_size'],'!=',self.config['fft_size'],'(vocoder)')
        if self.config['win_size'] != hparams['win_size']:
            print('Mismatch parameters: hparams[\'win_size\']=',hparams['win_size'],'!=',self.config['win_size'],'(vocoder)')
        if self.config['hop_size'] != hparams['hop_size']:
            print('Mismatch parameters: hparams[\'hop_size\']=',hparams['hop_size'],'!=',self.config['hop_size'],'(vocoder)')
        if self.config['fmin'] != hparams['fmin']:
            print('Mismatch parameters: hparams[\'fmin\']=',hparams['fmin'],'!=',self.config['fmin'] ,'(vocoder)')
        if self.config['fmax'] != hparams['fmax']:
            print('Mismatch parameters: hparams[\'fmax\']=',hparams['fmax'],'!=',self.config['fmax'] ,'(vocoder)')
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).transpose(2, 1).to(device)
            with utils.Timer('hifigan', print_time=hparams['profile_infer']):
                f0 = kwargs.get('f0')
                if f0 is not None and hparams.get('use_nsf'):
                    f0 = torch.FloatTensor(f0[None, :]).to(device)
                    y = self.model(c, f0).view(-1)
                else:
                    y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        if hparams.get('vocoder_denoise_c', 0.0) > 0:
            wav_out = denoise(wav_out, v=hparams['vocoder_denoise_c'])
        return wav_out

    # @staticmethod
    # def wav2spec(wav_fn, **kwargs):
    #     wav, _ = librosa.core.load(wav_fn, sr=hparams['audio_sample_rate'])
    #     wav_torch = torch.FloatTensor(wav)[None, :]
    #     mel = mel_spectrogram(wav_torch, hparams).numpy()[0]
    #     return wav, mel.T
