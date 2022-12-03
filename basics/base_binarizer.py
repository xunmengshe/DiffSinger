import shutil
from ast import Not
import os
os.environ["OMP_NUM_THREADS"] = "1"

from utils.multiprocess_utils import chunked_multiprocess_run
import random
import json
from resemblyzer import VoiceEncoder
from tqdm import tqdm
from data_gen.data_gen_utils import get_mel2ph, get_pitch_parselmouth, build_phone_encoder
from utils.hparams import set_hparams, hparams
from utils.phoneme_utils import build_phoneme_list
import numpy as np
from utils.indexed_datasets import IndexedDatasetBuilder


class BinarizationError(Exception):
    pass

BASE_ITEM_ATTRIBUTES = ['txt', 'ph', 'wav_fn', 'tg_fn', 'spk_id']

class BaseBinarizer:
    '''
        Base class for data processing.
        1. *process* and *process_data_split*:
            process entire data, generate the train-test split (support parallel processing);
        2. *process_item*:
            process singe piece of data;
        3. *get_pitch*:
            infer the pitch using some algorithm;
        4. *get_align*:
            get the alignment using 'mel2ph' format (see https://arxiv.org/abs/1905.09263).
        5. phoneme encoder, voice encoder, etc.

        Subclasses should define:
        1. *load_metadata*:
            how to read multiple datasets from files;
        2. *train_item_names*, *valid_item_names*, *test_item_names*:
            how to split the dataset;
        3. load_ph_set:
            the phoneme set.
    '''
    def __init__(self, processed_data_dir=None, item_attributes=BASE_ITEM_ATTRIBUTES):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']
        self.processed_data_dirs = processed_data_dir.split(",")
        self.binarization_args = hparams['binarization_args']
        self.pre_align_args = hparams['pre_align_args']
        
        self.items = {}
        # every item in self.items has some attributes
        self.item_attributes = item_attributes

        # load each dataset
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            self.load_meta_data(processed_data_dir, ds_id)
            if ds_id == 0:
                # check program correctness
                assert all([attr in self.item_attributes for attr in list(self.items.values())[0].keys()])
        self.item_names = sorted(list(self.items.keys()))
        
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        
        # set default get_pitch algorithm
        self.get_pitch_algorithm = get_pitch_parselmouth

    def load_meta_data(self, processed_data_dir, ds_id):
        raise NotImplementedError

    @property
    def train_item_names(self):
        raise NotImplementedError
        
    @property
    def valid_item_names(self):
        raise NotImplementedError

    @property
    def test_item_names(self):
        raise NotImplementedError

    def build_spk_map(self):
        spk_map = set()
        for item_name in self.item_names:
            spk_name = self.items[item_name]['spk_id']
            spk_map.add(spk_name)
        spk_map = {x: i for i, x in enumerate(sorted(list(spk_map)))}
        assert len(spk_map) == 0 or len(spk_map) <= hparams['num_spk'], len(spk_map)
        return spk_map

    def item_name2spk_id(self, item_name):
        return self.spk_map[self.items[item_name]['spk_id']]

    def _phone_encoder(self):
        ph_set = []
        # Just for ensuring the transcriptions match the dictionary.
        # May need refactoring in the future.
        dict_fn = os.path.join(hparams['binary_data_dir'], 'dictionary.txt')
        if hparams['reset_phone_dict'] or not os.path.exists(dict_fn):
            self.load_ph_set(ph_set)  # For singing, do checking and return the correct results.
            ph_set = sorted(set(ph_set))
            shutil.copy(hparams['g2p_dictionary'], dict_fn)
        else:
            ph_set = build_phoneme_list()
        return build_phone_encoder(ph_set)

    def load_ph_set(self, ph_set):
        raise NotImplementedError

    def meta_data_iterator(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            meta_data = self.items[item_name]
            yield item_name, meta_data

    def process(self):
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.spk_map = self.build_spk_map()
        print("| spk_map: ", self.spk_map)
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w', encoding='utf-8'))

        self.phone_encoder = self._phone_encoder()
        self.process_data_split('valid')
        self.process_data_split('test')
        self.process_data_split('train')

    def process_data_split(self, prefix, multiprocess=False):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        lengths = []
        f0s = []
        total_sec = 0
        if self.binarization_args['with_spk_embed']:
            voice_encoder = VoiceEncoder().cuda()

        for item_name, meta_data in self.meta_data_iterator(prefix):
            args.append([item_name, meta_data, self.binarization_args])
        
        if multiprocess:
            # code for parallel processing
            num_workers = int(os.getenv('N_PROC', os.cpu_count() // 3))
            for f_id, (_, item) in enumerate(
                    zip(tqdm(meta_data), chunked_multiprocess_run(self.process_item, args, num_workers=num_workers))):
                if item is None:
                    continue
                item['spk_embed'] = voice_encoder.embed_utterance(item['wav']) \
                    if self.binarization_args['with_spk_embed'] else None
                if not self.binarization_args['with_wav'] and 'wav' in item:
                    print("del wav")
                    del item['wav']
                builder.add_item(item)
                lengths.append(item['len'])
                total_sec += item['sec']
                if item.get('f0') is not None:
                    f0s.append(item['f0'])
        else:
            # code for single cpu processing
            for i in tqdm(reversed(range(len(args))), total=len(args)):
                a = args[i]
                item = self.process_item(*a)
                if item is None:
                    continue
                item['spk_embed'] = voice_encoder.embed_utterance(item['wav']) \
                    if self.binarization_args['with_spk_embed'] else None
                if not self.binarization_args['with_wav'] and 'wav' in item:
                    print("del wav")
                    del item['wav']
                builder.add_item(item)
                lengths.append(item['len'])
                total_sec += item['sec']
                if item.get('f0') is not None:
                    f0s.append(item['f0'])
        
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    def process_item(self, item_name, meta_data, binarization_args):
        from preprocessing.opencpop import File2Batch
        return File2Batch.temporary_dict2processed_input(item_name, meta_data, self.phone_encoder, binarization_args)

    def get_align(self, meta_data, mel, phone_encoded, res):
        raise NotImplementedError

    def get_align_from_textgrid(self, meta_data, mel, phone_encoded, res):
        '''
            NOTE: this part of script is *isolated* from other scripts, which means
                  it may not be compatible with the current version.
        '''
        return
        tg_fn, ph = meta_data['tg_fn'], meta_data['ph']
        if tg_fn is not None and os.path.exists(tg_fn):
            mel2ph, dur = get_mel2ph(tg_fn, ph, mel, hparams)
        else:
            raise BinarizationError(f"Align not found")
        if mel2ph.max() - 1 >= len(phone_encoded):
            raise BinarizationError(
                f"Align does not match: mel2ph.max() - 1: {mel2ph.max() - 1}, len(phone_encoded): {len(phone_encoded)}")
        res['mel2ph'] = mel2ph
        res['dur'] = dur

    def get_f0cwt(self, f0, res):
        '''
            NOTE: this part of script is *isolated* from other scripts, which means
                  it may not be compatible with the current version.
        '''
        return
        from utils.cwt import get_cont_lf0, get_lf0_cwt
        uv, cont_lf0_lpf = get_cont_lf0(f0)
        logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
        cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
        Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)
        if np.any(np.isnan(Wavelet_lf0)):
            raise BinarizationError("NaN CWT")
        res['cwt_spec'] = Wavelet_lf0
        res['cwt_scales'] = scales
        res['f0_mean'] = logf0s_mean_org
        res['f0_std'] = logf0s_std_org


if __name__ == "__main__":
    set_hparams()
    BaseBinarizer().process()
