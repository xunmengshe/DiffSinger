from typing import DefaultDict
from data_gen.singing.singing import SingingBinarizer, SINGING_ITEM_ATTRIBUTES
import os
import json
import numpy as np

from utils.hparams import hparams
from data_gen.tts.txt_processors.zh_g2pM import ALL_YUNMU

MIDISINGING_ITEM_ATTRIBUTES = SINGING_ITEM_ATTRIBUTES + ['pitch_midi', 'midi_dur', 'is_slur', 'ph_durs', 'word_boundary']

class MidiSingingBinarizer(SingingBinarizer):
    def __init__(self, processed_data_dir=None, item_attributes=MIDISINGING_ITEM_ATTRIBUTES):
        super().__init__(processed_data_dir, item_attributes)

    def load_meta_data(self, processed_data_dir, ds_id):
        # NOTE: this function is *isolated* from other scripts, which means
        # it may not be compatible with the current version.
        meta_midi = json.load(open(os.path.join(processed_data_dir, 'meta.json'), encoding='utf-8'))   # [list of dict]

        for song_item in meta_midi:
            item_name = raw_item_name = song_item['item_name']
            if len(self.processed_data_dirs) > 1:
                item_name = f'ds{ds_id}_{item_name}'
            
            item = {}
            item['wav_fn'] = song_item['wav_fn']
            item['txt'] = song_item['txt']

            item['ph'] = ' '.join(song_item['phs'])
            item['word boundary'] = [1 if x in ALL_YUNMU + ['AP', 'SP', '<SIL>'] else 0 for x in song_item['phs']]
            item['ph_durs'] = song_item['ph_dur']

            item['pitch_midi'] = song_item['notes']
            item['midi_dur'] = song_item['notes_dur']
            item['is_slur'] = song_item['is_slur']
            item['spk_id'] = 'pop-cs'
            if len(self.processed_data_dirs) > 1:
                self.item2spk[item_name] = f"ds{ds_id}_{self.item2spk[item_name]}"
            
            self.items[item_name] = item

    def get_align(self, meta_data, mel, phone_encoded, res, hop_size=hparams['hop_size'], audio_sample_rate=hparams['audio_sample_rate']):
        mel2ph = np.zeros([mel.shape[0]], int)
        startTime = 0
        ph_durs = meta_data['ph_durs']

        for i_ph in range(len(ph_durs)):
            start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
            end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
            mel2ph[start_frame:end_frame] = i_ph + 1
            startTime = startTime + ph_durs[i_ph]

        # print('ph durs: ', ph_durs)
        # print('mel2ph: ', mel2ph, len(mel2ph))
        res['mel2ph'] = mel2ph
        # res['dur'] = None
