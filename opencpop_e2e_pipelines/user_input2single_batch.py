'''
    user_input -> temporary_dict -> processed_input -> single_batch
'''
from utils.hparams import hparams
import torch
from pipelines import user_input2single_batch
import numpy as np
from modules.fastspeech.tts_modules import LengthRegulator
import librosa
from utils.text_encoder import TokenTextEncoder
from inference.opencpop.map import cpop_pinyin2ph_func
from pypinyin import lazy_pinyin

class UserInput2SingleBatch(user_input2single_batch.UserInput2SingleBatch):
    '''
        user_input -> temporary_dict -> processed_input -> single_batch
    '''

    @staticmethod
    def user_input2temporary_dict(user_inp, input_type):
        """

        :param user_inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """

        def preprocess_word_level_input(user_inp):
            # Pypinyin can't solve polyphonic words
            text_raw = user_inp['text'].replace('最长', '最常').replace('长睫毛', '常睫毛') \
                .replace('那么长', '那么常').replace('多长', '多常') \
                .replace('很长', '很常')  # We hope someone could provide a better g2p module for us by opening pull requests.

            # lyric
            pinyins = lazy_pinyin(text_raw, strict=False)
            pinyin2phs = cpop_pinyin2ph_func()
            ph_per_word_lst = [pinyin2phs[pinyin.strip()] for pinyin in pinyins if pinyin.strip() in pinyin2phs]

            # Note
            note_per_word_lst = [x.strip() for x in user_inp['notes'].split('|') if x.strip() != '']
            mididur_per_word_lst = [x.strip() for x in user_inp['notes_duration'].split('|') if x.strip() != '']

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
            return {'ph': ph_seq, 'notes': note_lst, 'midi_dur': midi_dur_lst, 'is_slur': is_slur}

        def preprocess_phoneme_level_input(user_inp):
            ph_seq = user_inp['ph_seq']
            note_lst = user_inp['note_seq'].split()
            midi_dur_lst = user_inp['note_dur_seq'].split()
            is_slur = [float(x) for x in user_inp['is_slur_seq'].split()]
            
            print(len(note_lst), len(ph_seq.split()), len(midi_dur_lst))
            if len(note_lst) == len(ph_seq.split()) == len(midi_dur_lst):
                print('Pass word-notes check.')
            else:
                print('The number of words does\'t match the number of notes\' windows. ',
                    'You should split the note(s) for each word by | mark.')
                return None
            
            # add support for user-defined phone duration
            ph_dur = None
            if user_inp['ph_dur'] is not None:
                ph_dur = np.array(user_inp['ph_dur'].split(),'float')
            else:
                print('Automatic phone duration mode')

            return {'ph': ph_seq, 'notes': note_lst, 'midi_dur': midi_dur_lst, 'is_slur': is_slur, 'ph_dur': ph_dur}

        # get ph seq, note lst, midi dur lst, is slur lst.
        if input_type == 'word':
            temp_dict = preprocess_word_level_input(user_inp)
        elif input_type == 'phoneme':  # like transcriptions.txt in Opencpop dataset.
            temp_dict = preprocess_phoneme_level_input(user_inp)
        else:
            print('Invalid input type.')
            return None

        return {**user_inp, **temp_dict}

    @staticmethod
    def temporary_dict2processed_input(temp_dict, ph_encoder):
        item_name = temp_dict.get('item_name', '<ITEM_NAME>')
        spk_name = temp_dict.get('spk_name', 'opencpop')

        # single spk
        spk_map = {'opencpop': 0}
        spk_id = spk_map[spk_name]
        
        try:
            ph_seq, note_lst, midi_dur_lst, is_slur = temp_dict['ph'], temp_dict['notes'], temp_dict['midi_dur'], temp_dict['is_slur']
            if 'ph_dur' in temp_dict:
                ph_dur = temp_dict['ph_dur']
            else:
                ph_dur = None
        except KeyError:
            print('==========> Preprocess_word_level or phone_level input wrong.')
            return None

        # convert note lst to midi id; convert note dur lst to midi duration
        try:
            midis = [librosa.note_to_midi(x.split("/")[0]) if x != 'rest' else 0
                     for x in note_lst]
            midi_dur_lst = [float(x) for x in midi_dur_lst]
        except Exception as e:
            print(e)
            print('Invalid Input Type.')
            return None

        ph_token = ph_encoder.encode(ph_seq)
        item = {'item_name': item_name, 'text': temp_dict['text'], 'ph': ph_seq, 'spk_id': spk_id,
                'ph_token': ph_token, 'pitch_midi': np.asarray(midis), 'midi_dur': np.asarray(midi_dur_lst),
                'is_slur': np.asarray(is_slur), 'ph_dur': ph_dur}
        item['ph_len'] = len(item['ph_token'])
        return item
    
    @staticmethod
    def processed_input2single_batch(processed_inp, device):
        item_names = [processed_inp['item_name']]
        text = [processed_inp['text']]
        ph = [processed_inp['ph']]
        txt_tokens = torch.LongTensor(processed_inp['ph_token'])[None, :].to(device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(device)
        spk_ids = torch.LongTensor(processed_inp['spk_id'])[None, :].to(device)

        pitch_midi = torch.LongTensor(processed_inp['pitch_midi'])[None, :hparams['max_frames']].to(device)
        midi_dur = torch.FloatTensor(processed_inp['midi_dur'])[None, :hparams['max_frames']].to(device)
        is_slur = torch.LongTensor(processed_inp['is_slur'])[None, :hparams['max_frames']].to(device)

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'spk_ids': spk_ids,
            'pitch_midi': pitch_midi,
            'midi_dur': midi_dur,
            'is_slur': is_slur
        }

        # feed user-defined phone duration to a module called LengthRegular
        # the output *mel2ph* will be used by FastSpeech
        # note: LengthRegular is not a neural network, only FastSpeech is. (see https://arxiv.org/abs/1905.09263)
        mel2ph = None
        if processed_inp['ph_dur'] is not None:
            ph_acc = np.around(np.add.accumulate(processed_inp['ph_dur']) * hparams['audio_sample_rate'] / hparams['hop_size'] + 0.5).astype('int')
            ph_dur = np.diff(ph_acc, prepend=0)
            ph_dur = torch.LongTensor(ph_dur)[None, :hparams['max_frames']].to(device)
            lr = LengthRegulator()
            mel2ph = lr(ph_dur, dur_padding=(batch['txt_tokens'] == 0)).detach()
        
        batch['mel2ph'] = mel2ph

        return batch
