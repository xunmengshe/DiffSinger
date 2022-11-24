import json
import os

from data_gen.singing import SingingBinarizer, SINGING_ITEM_ATTRIBUTES
from tts.data_gen.txt_processors.zh_g2pM import get_all_vowels

MIDISINGING_ITEM_ATTRIBUTES = SINGING_ITEM_ATTRIBUTES + ['pitch_midi', 'midi_dur', 'is_slur', 'ph_durs', 'word_boundary']


class MidiSingingBinarizer(SingingBinarizer):
    def __init__(self, processed_data_dir=None, item_attributes=MIDISINGING_ITEM_ATTRIBUTES):
        super().__init__(processed_data_dir, item_attributes)

    def load_meta_data(self, processed_data_dir, ds_id):
        '''
            NOTE: this function is *isolated* from other scripts, which means
                  it may not be compatible with the current version.
        '''
        return
        meta_midi = json.load(open(os.path.join(processed_data_dir, 'meta.json'), encoding='utf-8'))   # [list of dict]

        for song_item in meta_midi:
            item_name = raw_item_name = song_item['item_name']
            if len(self.processed_data_dirs) > 1:
                item_name = f'ds{ds_id}_{item_name}'
            
            item = {}
            item['wav_fn'] = song_item['wav_fn']
            item['txt'] = song_item['txt']

            item['ph'] = ' '.join(song_item['phs'])
            vowels = get_all_vowels()
            item['word boundary'] = [1 if x in vowels + ['AP', 'SP', '<SIL>'] else 0 for x in song_item['phs']]
            item['ph_durs'] = song_item['ph_dur']

            item['pitch_midi'] = song_item['notes']
            item['midi_dur'] = song_item['notes_dur']
            item['is_slur'] = song_item['is_slur']
            item['spk_id'] = 'pop-cs'
            if len(self.processed_data_dirs) > 1:
                self.item2spk[item_name] = f"ds{ds_id}_{self.item2spk[item_name]}"
            
            self.items[item_name] = item

