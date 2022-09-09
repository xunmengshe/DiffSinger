'''

    item: one piece of data
    item_name: data id
    wavfn: wave file path
    txt: lyrics
    ph: phoneme
    tgfn: text grid file path (unused)
    spk: dataset name
    wdb: word boundary
    ph_durs: phoneme durations
    midi: pitch as midi notes
    midi_dur: midi duration
    is_slur: keep singing upon note changes
'''

from collections import defaultdict
from data_gen.singing.midisinging import MidiSingingBinarizer
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
from basics.base_binarizer import BaseBinarizer, BinarizationError
from data_gen.tts.binarizer_zh import ZhBinarizer
from data_gen.tts.txt_processors.zh_g2pM import ALL_YUNMU
from src.vocoders.base_vocoder import VOCODERS

class OpencpopBinarizer(MidiSingingBinarizer):
    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)
        test_item_names = [x for x in item_names if any([x.startswith(ts) for ts in hparams['test_prefixes']])]
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    def load_meta_data(self, processed_data_dir, ds_id):
        del processed_data_dir, ds_id # unused

        raw_data_dir = hparams['raw_data_dir']
        # meta_midi = json.load(open(os.path.join(raw_data_dir, 'meta.json')))   # [list of dict]
        utterance_labels = open(os.path.join(raw_data_dir, 'transcriptions.txt'), encoding='utf-8').readlines()

        for utterance_label in utterance_labels:
            song_info = utterance_label.split('|')
            item_name = raw_item_name = song_info[0]
            item = {}

            item['wav_fn'] = f'{raw_data_dir}/wavs/{item_name}.wav'
            item['txt'] = song_info[1]

            item['ph'] = song_info[2]
            # self.item2wdb[item_name] = list(np.nonzero([1 if x in ALL_YUNMU + ['AP', 'SP'] else 0 for x in song_info[2].split()])[0])
            item['word_boundary'] = np.array([1 if x in ALL_YUNMU + ['AP', 'SP'] else 0 for x in song_info[2].split()])
            item['ph_durs'] = [float(x) for x in song_info[5].split(" ")]

            item['pitch_midi'] = np.array([librosa.note_to_midi(x.split("/")[0]) if x != 'rest' else 0
                                   for x in song_info[3].split(" ")])
            item['midi_dur'] = np.array([float(x) for x in song_info[4].split(" ")])
            item['is_slur'] = np.array([int(x) for x in song_info[6].split(" ")])
            item['spk_id'] = 'opencpop'
            assert item['pitch_midi'].shape == item['midi_dur'].shape == item['is_slur'].shape, \
                (item['pitch_midi'].shape, item['midi_dur'].shape, item['is_slur'].shape)

            self.items[item_name] = item
