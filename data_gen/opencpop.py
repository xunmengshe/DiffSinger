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
from data_gen.midisinging import MidiSingingBinarizer
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
from tts.data_gen.binarizer_zh import ZhBinarizer
from tts.data_gen.txt_processors.zh_g2pM import ALL_VOWELS
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
        from preprocessing.opencpop import File2Batch
        self.items = File2Batch.file2temporary_dict()