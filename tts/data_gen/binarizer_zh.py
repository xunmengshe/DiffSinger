import os

os.environ["OMP_NUM_THREADS"] = "1"

from tts.data_gen.txt_processors.zh_g2pM import get_all_consonants
from basics.base_binarizer import BaseBinarizer, BinarizationError
from data_gen.data_gen_utils import get_mel2ph
from utils.hparams import set_hparams, hparams
from utils.phoneme_utils import build_phoneme_list
import numpy as np
import pandas as pd

class ZhBinarizer(BaseBinarizer):
    def __init__(self, processed_data_dir=None):
        # use force_align
        self.forced_align = self.pre_align_args['forced_align']
        self.tg_dir = None
        if self.forced_align == 'mfa':
            self.tg_dir = 'mfa_outputs'
        if self.forced_align == 'kaldi':
            self.tg_dir = 'kaldi_outputs'
        
        super().__init__(processed_data_dir)
        
    def load_meta_data(self, processed_data_dir, ds_id):
        self.meta_df = pd.read_csv(f"{processed_data_dir}/metadata_phone.csv", dtype=str)
        for r_idx, r in self.meta_df.iterrows():
            item_name = raw_item_name = r['item_name']
            if len(self.processed_data_dirs) > 1:
                item_name = f'ds{ds_id}_{item_name}'
            item = {}
            item['txt'] = r['txt']
            item['ph'] = r['ph']
            item['wav_fn'] = os.path.join(hparams['raw_data_dir'], 'wavs', os.path.basename(r['wav_fn']).split('_')[1])
            item['spk_id'] = r.get('spk', 'SPK1')
            if len(self.processed_data_dirs) > 1:
                item['spk_id'] = f"ds{ds_id}_{self.item2spk[item_name]}"
            if self.tg_dir is not None:
                item['tg_fn'] = f"{processed_data_dir}/{self.tg_dir}/{raw_item_name}.TextGrid"
            self.items[item_name] = item

    def load_ph_set(self, ph_set):
        # load ph_set from pre-given dict
        ph_set += build_phoneme_list()

    @property
    def train_item_names(self):
        return self.item_names[hparams['test_num']+hparams['valid_num']:]
    
    @property
    def valid_item_names(self):
        return self.item_names[0: hparams['test_num']+hparams['valid_num']]  #

    @property
    def test_item_names(self):
        return self.item_names[0: hparams['test_num']]  # Audios for MOS testing are in 'test_ids'

    def get_align(self, tg_fn, ph, mel, phone_encoded, res):
        if tg_fn is not None and os.path.exists(tg_fn):
            _, dur = get_mel2ph(tg_fn, ph, mel, hparams)
        else:
            raise BinarizationError(f"Align not found")
        ph_list = ph.split(" ")
        assert len(dur) == len(ph_list)
        mel2ph = []
        # 分隔符的时长分配给韵母
        dur_cumsum = np.pad(np.cumsum(dur), [1, 0], mode='constant', constant_values=0)
        for i in range(len(dur)):
            p = ph_list[i]
            if p[0] != '<' and not p[0].isalpha():
                uv_ = res['f0'][dur_cumsum[i]:dur_cumsum[i + 1]] == 0
                j = 0
                while j < len(uv_) and not uv_[j]:
                    j += 1
                dur[i - 1] += j
                dur[i] -= j
                if dur[i] < 100:
                    dur[i - 1] += dur[i]
                    dur[i] = 0
        # 声母和韵母等长
        for i in range(len(dur)):
            p = ph_list[i]
            consonants = get_all_consonants()
            if p in consonants:
                p_next = ph_list[i + 1]
                if not (dur[i] > 0 and p_next[0].isalpha() and p_next not in consonants):
                    print(f"assert dur[i] > 0 and p_next[0].isalpha() and p_next not in ALL_SHENMU, "
                          f"dur[i]: {dur[i]}, p: {p}, p_next: {p_next}.")
                    continue
                total = dur[i + 1] + dur[i]
                dur[i] = total // 2
                dur[i + 1] = total - dur[i]
        for i in range(len(dur)):
            mel2ph += [i + 1] * dur[i]
        mel2ph = np.array(mel2ph)
        if mel2ph.max() - 1 >= len(phone_encoded):
            raise BinarizationError(f"| Align does not match: {(mel2ph.max() - 1, len(phone_encoded))}")
        res['mel2ph'] = mel2ph
        res['dur'] = dur


if __name__ == "__main__":
    set_hparams()
    ZhBinarizer().process()
