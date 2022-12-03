# coding=utf8
import os

import torch
import numpy as np
from src.vocoders.base_vocoder import VOCODERS

from utils import load_ckpt
from utils.hparams import set_hparams, hparams
import librosa
import glob
import re
from utils.phoneme_utils import build_g2p_dictionary, build_phoneme_list
from utils.text_encoder import TokenTextEncoder
from pypinyin import pinyin, lazy_pinyin, Style


class BaseSVSInfer:
    '''
        Base class for SVS inference models.
        1. *example_run* and *infer_once*:
            the overall pipeline;
        2. *build_vocoder* and *run_vocoder*:
            a HifiGAN vocoder;
        3. *preprocess_word_level_input*:
            convert words to phonemes, add slurs.

        Subclasses should define:
        1. *build_model*:
            how to build the model;
        2. *forward_model*:
            how to run the model (typically, generate a mel-spectrogram and
            pass it to the pre-built vocoder);
        3. *preprocess_input*:
            how to preprocess user input.
    '''
    def __init__(self, hparams, device=None, load_model=True, load_vocoder=True):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.device = device

        if load_model:
            phone_list = build_phoneme_list()
            self.ph_encoder = TokenTextEncoder(vocab_list=phone_list, replace_oov=',')
            self.pinyin2phs = build_g2p_dictionary()
            self.spk_map = {'opencpop': 0}
            self.model = self.build_model()
            self.model.eval()
            self.model.to(self.device)
        if load_vocoder:
            self.vocoder = self.build_vocoder()
            self.vocoder.model.eval()
            self.vocoder.model.to(self.device)

    def build_model(self):
        raise NotImplementedError

    def forward_model(self, inp, return_mel):
        raise NotImplementedError

    def build_vocoder(self):
        if hparams['vocoder'] in VOCODERS:
            vocoder = VOCODERS[hparams['vocoder']]()
        else:
            vocoder = VOCODERS[hparams['vocoder'].split('.')[-1]]()
        return vocoder

    def run_vocoder(self, c, **kwargs):
        y = self.vocoder.spec2wav_torch(c,**kwargs)
        return y[None]

    def preprocess_word_level_input(self, inp):
        # Pypinyin can't solve polyphonic words
        text_raw = inp['text'].replace('最长', '最常').replace('长睫毛', '常睫毛') \
            .replace('那么长', '那么常').replace('多长', '多常') \
            .replace('很长', '很常')  # We hope someone could provide a better g2p module for us by opening pull requests.

        # lyric
        pinyins = lazy_pinyin(text_raw, strict=False)
        ph_per_word_lst = [' '.join(self.pinyin2phs[pinyin.strip()])
                           for pinyin in pinyins
                           if pinyin.strip() in self.pinyin2phs]

        # Note
        note_per_word_lst = [x.strip() for x in inp['notes'].split('|') if x.strip() != '']
        mididur_per_word_lst = [x.strip() for x in inp['notes_duration'].split('|') if x.strip() != '']

        if len(note_per_word_lst) == len(ph_per_word_lst) == len(mididur_per_word_lst):
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
            print(ph_per_word_lst, note_per_word_lst, mididur_per_word_lst)
            print(len(ph_per_word_lst), len(note_per_word_lst), len(mididur_per_word_lst))
            return None

        note_lst = []
        ph_lst = []
        midi_dur_lst = []
        is_slur = []
        for idx, ph_per_word in enumerate(ph_per_word_lst):
            # for phs in one word:
            # single ph like ['ai']  or multiple phs like ['n', 'i']
            ph_in_this_word = ph_per_word.split()

            # for notes in one word:
            # single note like ['D4'] or multiple notes like ['D4', 'E4'] which means a 'slur' here.
            note_in_this_word = note_per_word_lst[idx].split()
            midi_dur_in_this_word = mididur_per_word_lst[idx].split()
            # process for the model input
            # Step 1.
            #  Deal with note of 'not slur' case or the first note of 'slur' case
            #  j        ie
            #  F#4/Gb4  F#4/Gb4
            #  0        0
            for ph in ph_in_this_word:
                ph_lst.append(ph)
                note_lst.append(note_in_this_word[0])
                midi_dur_lst.append(midi_dur_in_this_word[0])
                is_slur.append(0)
            # step 2.
            #  Deal with the 2nd, 3rd... notes of 'slur' case
            #  j        ie         ie
            #  F#4/Gb4  F#4/Gb4    C#4/Db4
            #  0        0          1
            if len(note_in_this_word) > 1:  # is_slur = True, we should repeat the YUNMU to match the 2nd, 3rd... notes.
                for idx in range(1, len(note_in_this_word)):
                    ph_lst.append(ph_in_this_word[-1])
                    note_lst.append(note_in_this_word[idx])
                    midi_dur_lst.append(midi_dur_in_this_word[idx])
                    is_slur.append(1)
        ph_seq = ' '.join(ph_lst)

        if len(ph_lst) == len(note_lst) == len(midi_dur_lst):
            print(len(ph_lst), len(note_lst), len(midi_dur_lst))
            print('Pass word-notes check.')
        else:
            print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
            return None
        return ph_seq, note_lst, midi_dur_lst, is_slur

    def preprocess_input(self, inp, input_type):
        raise NotImplementedError

    def postprocess_output(self, output):
        return output

    def infer_once(self, inp, return_mel=False):
        inp = self.preprocess_input(inp, input_type=inp['input_type'] if inp.get('input_type') else 'word')
        output = self.forward_model(inp, return_mel=return_mel)
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
