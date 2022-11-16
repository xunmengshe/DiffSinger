import glob
import os.path
import re

import matplotlib.pyplot as plt

from basics.base_binarizer import BaseBinarizer, BASE_ITEM_ATTRIBUTES
from utils.hparams import hparams
from utils.phoneme_utils import build_phoneme_list

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
            item['spk_id'] = re.split('[-#]', piece_path.split('/')[-2])[0]
            item['tg_fn'] = piece_path.replace(wav_suffix, tg_suffix)
            item_name = piece_path[len(processed_data_dir)+1:].replace('/', '-')[:-len(wav_suffix)]
            if len(self.processed_data_dirs) > 1:
                item_name = f'ds{ds_id}_{item_name}'
                item['spk_id'] = f"ds{ds_id}_{item['spk_id']}"
            
            self.items[item_name] = item

    def generate_summary(self, phone_set: set):
        # Group by phonemes.
        phoneme_map = {}
        for ph in sorted(phone_set):
            phoneme_map[ph] = 0
        for item in self.items.values():
            for ph, slur in zip(item['ph'].split(), item['is_slur']):
                if ph not in phone_set or slur == 1:
                    continue
                phoneme_map[ph] += 1
        # Draw graph.
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(int(len(phone_set) * 0.8), 10))
        x = list(phoneme_map.keys())
        values = list(phoneme_map.values())
        plt.bar(x=x, height=values)
        plt.tick_params(labelsize=15)
        plt.xlim(-1, len(phone_set))
        for a, b in zip(x, values):
            plt.text(a, b + 2, b, ha='center', va='bottom', fontsize=15)
        plt.grid()
        plt.title('Phoneme Distribution Summary', fontsize=30)
        plt.xlabel('Phoneme', fontsize=20)
        plt.ylabel('Number of occurrences', fontsize=20)
        plt.savefig(fname=os.path.join(hparams['binary_data_dir'], 'phoneme_distribution.jpg'),
                    bbox_inches='tight',
                    pad_inches=0.25)
        print('===== Phoneme Distribution Summary =====')
        for i, key in enumerate(sorted(phoneme_map.keys())):
            if i == len(phone_set) - 1:
                end = '\n'
            elif i % 15 == 14:
                end = ',\n'
            else:
                end = ', '
            print(f'\'{key}\': {phoneme_map[key]}', end=end)

    def load_ph_set(self, ph_set):
        # load those phones that appear in the actual data
        for item in self.items.values():
            ph_set += item['ph'].split(' ')
        # check unrecognizable or missing phones
        actual_phone_set = set(ph_set)
        required_phone_set = set(build_phoneme_list())
        self.generate_summary(required_phone_set)
        if actual_phone_set != required_phone_set:
            unrecognizable_phones = actual_phone_set.difference(required_phone_set)
            missing_phones = required_phone_set.difference(actual_phone_set)
            raise AssertionError('transcriptions and dictionary mismatch.\n'
                                 f' (+) {sorted(unrecognizable_phones)}\n'
                                 f' (-) {sorted(missing_phones)}')


if __name__ == "__main__":
    # NOTE: this line is *isolated* from other scripts, which means
    # it may not be compatible with the current version.
    SingingBinarizer().process()
