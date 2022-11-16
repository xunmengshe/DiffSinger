import os
import random
from copy import deepcopy
import pandas as pd
import logging
from tqdm import tqdm
import json
import glob
import re
from resemblyzer import VoiceEncoder
import traceback
import numpy as np
import pretty_midi
import librosa
from scipy.interpolate import interp1d
import torch
from textgrid import TextGrid

from utils.hparams import hparams
from data_gen.data_gen_utils import build_phone_encoder, get_pitch_parselmouth
from utils.pitch_utils import f0_to_coarse
from basics.base_binarizer import BaseBinarizer, BinarizationError, BASE_ITEM_ATTRIBUTES
from tts.data_gen.binarizer_zh import ZhBinarizer
from tts.data_gen.txt_processors.zh_g2pM import ALL_VOWELS
from src.vocoders.base_vocoder import VOCODERS

SINGING_ITEM_ATTRIBUTES = BASE_ITEM_ATTRIBUTES + ['f0_fn']

class SingingBinarizer(BaseBinarizer):
    def __init__(self, processed_data_dir=None, item_attributes=SINGING_ITEM_ATTRIBUTES):
        super().__init__(processed_data_dir, item_attributes)

        print('spkers: ', set(item['spk_id'] for item in self.items.values()))
        self.item_names = sorted(list(self.items.keys()))
        self._train_item_names, self._test_item_names = self.split_train_test_set(self.item_names)

    def split_train_test_set(self, item_names):
        raise NotImplementedError

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names

    @property
    def test_item_names(self):
        return self._test_item_names

    def load_meta_data(self, processed_data_dir, ds_id):
        wav_suffix = '_wf0.wav'
        txt_suffix = '.txt'
        ph_suffix = '_ph.txt'
        tg_suffix = '.TextGrid'
        all_wav_pieces = glob.glob(f'{processed_data_dir}/*/*{wav_suffix}')

        for piece_path in all_wav_pieces:
            item = {}

            item['txt'] = open(f'{piece_path.replace(wav_suffix, txt_suffix)}', encoding='utf-8').readline()
            item['ph'] = open(f'{piece_path.replace(wav_suffix, ph_suffix)}', encoding='utf-8').readline()
            item['wav_fn'] = piece_path
            item['spk_id'] = re.split('-|#', piece_path.split('/')[-2])[0]
            item['tg_fn'] = piece_path.replace(wav_suffix, tg_suffix)
            item_name = piece_path[len(processed_data_dir)+1:].replace('/', '-')[:-len(wav_suffix)]
            if len(self.processed_data_dirs) > 1:
                item_name = f'ds{ds_id}_{item_name}'
                item['spk_id'] = f"ds{ds_id}_{item['spk_id']}"
            
            self.items[item_name] = item

    def load_ph_set(self, ph_set):
        # load those phones that appear in the actual data
        for item in self.items.values():
            ph_set += item['ph'].split(' ')

if __name__ == "__main__":
    # NOTE: this line is *isolated* from other scripts, which means
    # it may not be compatible with the current version.
    SingingBinarizer().process()
