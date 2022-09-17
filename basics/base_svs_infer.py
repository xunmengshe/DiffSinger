'''
    Base class for SVS inference models.
    1. *example_run* and *infer_once*:
        the overall pipeline;
    2. *build_vocoder* and *run_vocoder*:
        a HifiGAN vocoder.

    Subclasses should define:
    1. *build_model*:
        how to build the model;
    2. *forward_model*:
        how to run the model (typically, generate a mel-spectrogram and
        pass it to the pre-built vocoder);
    3. *preprocess_input*:
        how to preprocess user input.
'''

# coding=utf8
import os

import torch
import numpy as np
from modules.hifigan.hifigan import HifiGanGenerator
from src.vocoders.hifigan import HifiGAN

from utils import load_ckpt
from utils.hparams import set_hparams, hparams
import librosa
import glob
import re
from utils.text_encoder import TokenTextEncoder


class BaseSVSInfer:
    def __init__(self, hparams, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.device = device

        phone_list = ["AP", "SP", "a", "ai", "an", "ang", "ao", "b", "c", "ch", "d", "e", "ei", "en", "eng", "er", "f", "g",
                "h", "i", "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong", "iu", "j", "k", "l", "m", "n", "o",
                "ong", "ou", "p", "q", "r", "s", "sh", "t", "u", "ua", "uai", "uan", "uang", "ui", "un", "uo", "v",
                "van", "ve", "vn", "w", "x", "y", "z", "zh"]
        self.ph_encoder = TokenTextEncoder(None, vocab_list=phone_list, replace_oov=',')

        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        self.vocoder = self.build_vocoder()
        self.vocoder.eval()
        self.vocoder.to(self.device)

    def build_model(self):
        raise NotImplementedError

    def forward_model(self, inp):
        raise NotImplementedError

    def build_vocoder(self):
        base_dir = hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        file_path = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.*'), key=
        lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).*', x.replace('\\','/'))[0]))[-1]
        print('| load HifiGAN: ', file_path)
        ext = os.path.splitext(file_path)[-1]
        if ext == '.pth':
            vocoder = torch.load(file_path, map_location="cpu")
        elif ext == '.ckpt':
            ckpt_dict = torch.load(file_path, map_location="cpu")
            config = set_hparams(config_path, global_hparams=False)
            state = ckpt_dict["state_dict"]["model_gen"]
            vocoder = HifiGanGenerator(config)
            vocoder.load_state_dict(state, strict=True)
            vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(self.device)
        return vocoder

    def run_vocoder(self, c, **kwargs):
        c = c.transpose(2, 1)  # [B, 80, T]
        f0 = kwargs.get('f0')  # [B, T]
        if f0 is not None and hparams.get('use_nsf'):
            # f0 = torch.FloatTensor(f0).to(self.device)
            y = self.vocoder(c, f0).view(-1)
        else:
            y = self.vocoder(c).view(-1)
            # [T]
        return y[None]

    def preprocess_input(self):
        raise NotImplementedError

    def postprocess_output(self, output):
        return output

    def infer_once(self, inp):
        inp = self.preprocess_input(inp, input_type=inp['input_type'] if inp.get('input_type') else 'word')
        output = self.forward_model(inp)
        output = self.postprocess_output(output)
        return output

    @classmethod
    def example_run(cls, inp, target='infer_out/example_out.wav'):
        # settings hparams
        set_hparams(print_hparams=False)
        
        # call the model
        infer_ins = cls(hparams)
        out = infer_ins.infer_once(inp)
        
        # output to file
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print(f'| save audio: {target}')
        from utils.audio import save_wav
        save_wav(out, target, hparams['audio_sample_rate'])


# if __name__ == '__main__':
    # debug
    # a = BaseSVSInfer(hparams)
    # a.preprocess_input({'text': '你 说 你 不 SP 懂 为 何 在 这 时 牵 手 AP',
    #                     'notes': 'D#4/Eb4 | D#4/Eb4 | D#4/Eb4 | D#4/Eb4 | rest | D#4/Eb4 | D4 | D4 | D4 | D#4/Eb4 | F4 | D#4/Eb4 | D4 | rest',
    #                     'notes_duration': '0.113740 | 0.329060 | 0.287950 | 0.133480 | 0.150900 | 0.484730 | 0.242010 | 0.180820 | 0.343570 | 0.152050 | 0.266720 | 0.280310 | 0.633300 | 0.444590'
    #                     })

    # b = {
    #     'text': '小酒窝长睫毛AP是你最美的记号',
    #     'notes': 'C#4/Db4 | F#4/Gb4 | G#4/Ab4 | A#4/Bb4 F#4/Gb4 | F#4/Gb4 C#4/Db4 | C#4/Db4 | rest | C#4/Db4 | A#4/Bb4 | G#4/Ab4 | A#4/Bb4 | G#4/Ab4 | F4 | C#4/Db4',
    #     'notes_duration': '0.407140 | 0.376190 | 0.242180 | 0.509550 0.183420 | 0.315400 0.235020 | 0.361660 | 0.223070 | 0.377270 | 0.340550 | 0.299620 | 0.344510 | 0.283770 | 0.323390 | 0.360340'
    # }
    # c = {
    #     'text': '小酒窝长睫毛AP是你最美的记号',
    #     'ph_seq': 'x iao j iu w o ch ang ang j ie ie m ao AP sh i n i z ui m ei d e j i h ao',
    #     'note_seq': 'C#4/Db4 C#4/Db4 F#4/Gb4 F#4/Gb4 G#4/Ab4 G#4/Ab4 A#4/Bb4 A#4/Bb4 F#4/Gb4 F#4/Gb4 F#4/Gb4 C#4/Db4 C#4/Db4 C#4/Db4 rest C#4/Db4 C#4/Db4 A#4/Bb4 A#4/Bb4 G#4/Ab4 G#4/Ab4 A#4/Bb4 A#4/Bb4 G#4/Ab4 G#4/Ab4 F4 F4 C#4/Db4 C#4/Db4',
    #     'note_dur_seq': '0.407140 0.407140 0.376190 0.376190 0.242180 0.242180 0.509550 0.509550 0.183420 0.315400 0.315400 0.235020 0.361660 0.361660 0.223070 0.377270 0.377270 0.340550 0.340550 0.299620 0.299620 0.344510 0.344510 0.283770 0.283770 0.323390 0.323390 0.360340 0.360340',
    #     'is_slur_seq': '0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    # }  # input like Opencpop dataset.
    # a.preprocess_input(b)
    # a.preprocess_input(c, input_type='phoneme')
