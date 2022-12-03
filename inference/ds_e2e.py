import torch
# from inference.tts.fs import FastSpeechInfer
# from modules.tts.fs2_orig import FastSpeech2Orig
from basics.base_svs_infer import BaseSVSInfer
from utils import load_ckpt
from utils.hparams import hparams
from src.diff.diffusion import GaussianDiffusion
from src.diffsinger_task import DIFF_DECODERS
from modules.fastspeech.pe import PitchExtractor
import utils
from modules.fastspeech.tts_modules import LengthRegulator
import librosa
import numpy as np


class DiffSingerE2EInfer(BaseSVSInfer):
    def build_model(self):
        model = GaussianDiffusion(
            phone_encoder=self.ph_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )

        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model')

        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            self.pe = PitchExtractor().to(self.device)
            utils.load_ckpt(self.pe, hparams['pe_ckpt'], 'model', strict=True)
            self.pe.eval()
        return model

    def preprocess_word_level_input(self, inp):
        return super().preprocess_word_level_input(inp)

    def preprocess_phoneme_level_input(self, inp):
        ph_seq = inp['ph_seq']
        note_lst = inp['note_seq'].split()
        midi_dur_lst = inp['note_dur_seq'].split()
        is_slur = np.array(inp['is_slur_seq'].split(),'float')
        ph_dur = None
        if inp['ph_dur'] is not None:
            ph_dur = np.array(inp['ph_dur'].split(),'float')
            print(len(note_lst), len(ph_seq.split()), len(midi_dur_lst), len(ph_dur))
            if len(note_lst) == len(ph_seq.split()) == len(midi_dur_lst) == len(ph_dur):
                print('Pass word-notes check.')
            else:
                print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
                return None
        else:
            print('Automatic phone duration mode')
            print(len(note_lst), len(ph_seq.split()), len(midi_dur_lst))
            if len(note_lst) == len(ph_seq.split()) == len(midi_dur_lst):
                print('Pass word-notes check.')
            else:
                print('The number of words does\'t match the number of notes\' windows. ',
                  'You should split the note(s) for each word by | mark.')
                return None
        return ph_seq, note_lst, midi_dur_lst, is_slur, ph_dur
        
    def preprocess_input(self, inp, input_type='word'):
        """

        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """

        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', 'opencpop')

        # single spk
        spk_id = self.spk_map[spk_name]

        # get ph seq, note lst, midi dur lst, is slur lst.
        if input_type == 'word':
            ret = self.preprocess_word_level_input(inp)
        elif input_type == 'phoneme':  # like transcriptions.txt in Opencpop dataset.
            ret = self.preprocess_phoneme_level_input(inp)
        else:
            print('Invalid input type.')
            return None

        if ret:
            if input_type == 'word':
                ph_seq, note_lst, midi_dur_lst, is_slur = ret
            else:
                ph_seq, note_lst, midi_dur_lst, is_slur, ph_dur = ret
        else:
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

        ph_token = self.ph_encoder.encode(ph_seq)
        item = {'item_name': item_name, 'text': inp['text'], 'ph': ph_seq, 'spk_id': spk_id,
                'ph_token': ph_token, 'pitch_midi': np.asarray(midis), 'midi_dur': np.asarray(midi_dur_lst),
                'is_slur': np.asarray(is_slur), 'ph_dur': None}
        item['ph_len'] = len(item['ph_token'])
        if input_type == 'phoneme' :
            item['ph_dur'] = ph_dur
        return item
    
    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(self.device)

        pitch_midi = torch.LongTensor(item['pitch_midi'])[None, :hparams['max_frames']].to(self.device)
        midi_dur = torch.FloatTensor(item['midi_dur'])[None, :hparams['max_frames']].to(self.device)
        is_slur = torch.LongTensor(item['is_slur'])[None, :hparams['max_frames']].to(self.device)
        mel2ph = None
        if item['ph_dur'] is not None:
            ph_acc=np.around(np.add.accumulate(item['ph_dur'])*hparams['audio_sample_rate']/hparams['hop_size']+0.5).astype('int')
            ph_dur=np.diff(ph_acc,prepend=0)
            ph_dur = torch.LongTensor(ph_dur)[None, :hparams['max_frames']].to(self.device)
            lr=LengthRegulator()
            mel2ph=lr(ph_dur,txt_tokens==0).detach()

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'spk_ids': spk_ids,
            'pitch_midi': pitch_midi,
            'midi_dur': midi_dur,
            'is_slur': is_slur,
            'mel2ph': mel2ph
        }
        return batch
        
    def forward_model(self, inp, return_mel=False):
        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_id = sample.get('spk_ids')
        with torch.no_grad():
            output = self.model(txt_tokens, spk_id=spk_id, ref_mels=None, infer=True,
                                pitch_midi=sample['pitch_midi'], midi_dur=sample['midi_dur'],
                                is_slur=sample['is_slur'],mel2ph=sample['mel2ph'])
            mel_out = output['mel_out']  # [B, T,80]
            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                f0_pred = self.pe(mel_out)['f0_denorm_pred']  # pe predict from Pred mel
            else:
                f0_pred = output['f0_denorm']
            if return_mel:
                return mel_out.cpu(), f0_pred.cpu()
            wav_out = self.run_vocoder(mel_out, f0=f0_pred)
        wav_out = wav_out.cpu().numpy()
        return wav_out[0]


if __name__ == '__main__':
    inp1 = {
        'text': 'SP一闪一闪亮晶晶SP满天都是小星星',
        'notes': 'rest|C4|C4|G4|G4|A4|A4|G4|rest|F4|F4|E4|E4|D4|D4|C4',
        'notes_duration': '1|0.5|0.5|0.5|0.5|0.5|0.5|0.75|0.25|0.5|0.5|0.5|0.5|0.5|0.5|0.75',
        'input_type': 'word' # Automatic phone duration mode
    }  # user input: Chinese characters
    inp2 = {
        'text': 'SP 好 一 朵 美 丽 地 茉 莉 花 SP 好 一 朵 美 丽 地 茉 莉 花 SP 芬 芳 美 丽 满 枝 芽 SP 又 香 又 白 人 人 夸 SP 让 我 来 将 你 摘 下 SP 送 给 别 人 家 SP 茉 莉 花 呀 茉 莉 花 SP',
        'ph_seq': 'SP h ao y i d uo m ei ei l i d i m o l i i h ua SP h ao y i d uo m ei ei l i d i m o l i i h ua SP f en f ang m ei l i i m an zh i y a SP y ou x iang iang y ou b ai ai r en r en en k ua SP r ang ang w o o l ai j iang n i zh ai ai x ia SP s ong g ei ei b ie ie r en en j ia SP m o l i h ua y a m o o l i i h ua SP',
        'note_seq': 'rest E4 E4 E4 E4 G4 G4 A4 A4 C5 C5 C5 A4 A4 G4 G4 G4 G4 A4 G4 G4 rest E4 E4 E4 E4 G4 G4 A4 A4 C5 C5 C5 A4 A4 G4 G4 G4 G4 A4 G4 G4 rest G4 G4 G4 G4 G4 G4 E4 E4 G4 A4 A4 A4 A4 G4 G4 rest E4 E4 D4 D4 E4 G4 G4 E4 E4 D4 C4 C4 C4 C4 D4 C4 C4 rest E4 E4 D4 C4 C4 E4 D4 D4 E4 E4 G4 G4 A4 A4 C5 G4 G4 rest D4 D4 E4 E4 G4 C4 C4 D4 C4 C4 A3 G3 G3 rest A3 A3 C4 C4 D4 D4 E4 E4 C4 C4 D4 C4 C4 A3 G3 G3 rest',
        'note_dur_seq': '1 0.7058824 0.7058824 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.7058824 0.7058824 0.3529412 0.3529412 0.3529412 1.058824 1.058824 0.352941 0.7058824 0.7058824 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.7058824 0.7058824 0.3529412 0.3529412 0.3529412 1.058824 1.058824 0.352941 0.7058824 0.7058824 0.7058824 0.7058824 0.7058824 0.7058824 0.3529412 0.3529412 0.3529412 0.7058824 0.7058824 0.7058824 0.7058824 1.058824 1.058824 0.352941 0.7058824 0.7058824 0.3529412 0.3529412 0.3529412 0.7058824 0.7058824 0.3529412 0.3529412 0.3529412 0.7058824 0.7058824 0.3529412 0.3529412 0.3529412 1.058824 1.058824 0.352941 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 1.058824 1.058824 0.3529412 0.3529412 0.7058824 0.7058824 0.3529412 0.3529412 0.3529412 1.058824 1.058824 0.352941 0.7058824 0.7058824 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 1.058824 1.058824 0.352941 0.7058824 0.7058824 0.7058824 0.7058824 1.058824 1.058824 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 0.3529412 1.058824 1.058824 1',
        'is_slur_seq': '0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0',
        'ph_dur': None,  # Automatic phone duration mode
        'input_type': 'phoneme'
    }  # input like Opencpop dataset.
    inp3 = {
        'text': 'SP 还 记 得 那 场 音 乐 会 的 烟 火 SP 还 记 得 那 个 凉 凉 的 深 秋 SP 还 记 得 人 潮 把 你 推 向 了 我 SP 游 乐 园 拥 挤 的 正 是 时 候 SP 一 个 夜 晚 坚 持 不 睡 的 等 候 SP 一 起 泡 温 泉 奢 侈 的 享 受 SP 有 一 次 日 记 里 愚 蠢 的 困 惑 SP 因 为 你 的 微 笑 幻 化 成 风 SP 你 大 大 的 勇 敢 保 护 着 我 SP 我 小 小 的 关 怀 喋 喋 不 休 SP 感 谢 我 们 一 起 走 了 那 么 久 SP 又 再 一 次 回 到 凉 凉 深 秋 SP 给 你 我 的 手 SP 像 温 柔 野 兽 SP 把 自 由 交 给 草 原 的 辽 阔 SP 我 们 小 手 拉 大 手 SP 一 起 郊 游 SP 今 天 别 想 太 多 SP 你 是 我 的 梦 SP 像 北 方 的 风 SP 吹 着 南 方 暖 洋 洋 的 哀 愁 SP 我 们 小 手 拉 大 手 SP 今 天 加 油 SP 向 昨 天 挥 挥 手 SP',
        'ph_seq': 'SP h ai j i d e n a ch ang y in y ve h ui d e y an h uo uo SP h ai j i d e n a g e l iang l iang d e sh en en q iu iu SP h ai j i d e r en ch ao b a n i t ui x iang l e w o o SP y ou l e y van y ong j i d e zh eng sh i sh i h ou ou SP y i g e y e w an j ian ch i b u sh ui d e d eng h ou ou SP y i q i p ao w en q van sh e ch i d e x iang iang sh ou ou SP y ou y i c i r i j i l i y v ch un d e k un h uo uo SP y in w ei n i d e w ei x iao h uan h ua ch eng f eng eng SP n i d a d a d e y ong g an b ao h u zh e w o o SP w o x iao x iao d e g uan h uai d ie d ie b u x iu iu SP g an x ie w o m en y i q i z ou l e n a m e j iu iu SP y ou z ai y i c i h ui d ao ao l iang l iang sh en q iu iu SP g ei n i w o d e sh ou SP x iang w en r ou y e sh ou SP b a z i y ou j iao g ei c ao y van d e l iao iao k uo uo uo SP w o m en x iao sh ou l a d a sh ou SP y i q i j iao iao y ou SP j in t ian b ie x iang t ai d uo uo SP n i sh i w o d e m eng SP x iang b ei f ang d e f eng SP ch ui zh e n an f ang n uan y ang y ang d e ai ai ch ou ou ou SP w o m en x iao sh ou l a d a sh ou SP j in t ian j ia ia y ou SP x iang z uo t ian h ui h ui ui sh ou ou ou SP',
        'note_seq': 'rest G3 G3 G3 G3 A3 A3 C4 C4 D4 D4 E4 E4 A4 A4 G4 G4 E4 E4 D4 D4 D4 D4 C4 rest C4 C4 D4 D4 C4 C4 B3 B3 C4 C4 F4 F4 A3 A3 C4 C4 D4 D4 E4 E4 E4 D4 rest D4 D4 E4 E4 D4 D4 C#4 C#4 D4 D4 G4 G4 B3 B3 D4 D4 E4 E4 D4 D4 D4 D4 C4 rest C4 C4 D4 D4 C4 C4 B3 B3 C4 C4 F4 F4 A3 A3 C4 C4 A3 A3 A3 A3 G3 rest G3 G3 G3 G3 A3 A3 C4 C4 D4 D4 E4 E4 A4 A4 G4 G4 E4 E4 D4 D4 D4 D4 C4 rest C4 C4 D4 D4 C4 C4 B3 B3 C4 C4 F4 F4 A3 A3 C4 C4 D4 D4 E4 E4 E4 D4 rest D4 D4 E4 E4 D4 D4 C#4 C#4 D4 D4 G4 G4 B3 B3 D4 D4 E4 E4 D4 D4 D4 D4 C4 rest C4 C4 D4 D4 C4 C4 B3 B3 C4 C4 F4 F4 A3 A3 C4 C4 D4 D4 D4 D4 C4 rest E4 E4 F4 F4 E4 E4 D4 D4 E4 E4 F4 F4 E4 E4 D4 D4 E4 E4 E4 E4 F4 rest F4 F4 G4 G4 F4 F4 G4 G4 F4 F4 E4 E4 D4 D4 C4 C4 D4 D4 D4 D4 E4 rest E4 E4 E4 E4 D4 D4 C#4 C#4 E4 E4 E4 E4 D4 D4 D4 D4 D4 D4 C#4 C#4 C#4 C#4 D4 rest D4 D4 D4 D4 E4 E4 F#4 F#4 D4 D4 G4 G4 A4 G4 G4 G4 G4 F#4 F#4 F#4 F#4 G4 rest E4 E4 F4 F4 E4 E4 F4 F4 G4 G4 rest E4 E4 F4 F4 E4 E4 F4 F4 G4 G4 rest G4 G4 A4 A4 G4 G4 A4 A4 B4 B4 C5 C5 E4 E4 E4 E4 G4 G4 A4 A4 A4 G4 G4 rest C4 C4 D4 D4 C4 C4 F4 F4 E4 E4 D4 D4 C4 C4 rest F4 F4 E4 E4 D4 D4 C4 C4 C4 rest C4 C4 D4 D4 A3 A3 C4 C4 E4 E4 E4 E4 G4 rest E4 E4 F4 F4 E4 E4 F4 F4 G4 G4 rest E4 E4 F4 F4 E4 E4 F4 F4 G4 G4 rest G4 G4 A4 A4 G4 G4 A4 A4 B4 B4 C5 C5 E4 E4 E4 E4 G4 A4 A4 A4 G4 G4 rest C4 C4 D4 D4 C4 C4 F4 F4 E4 E4 D4 D4 C4 C4 rest F4 F4 E4 E4 D4 D4 C4 C4 C4 rest C4 C4 D4 D4 A3 A3 C4 C4 C4 C4 D4 D4 D4 C4 C4 rest',
        'note_dur_seq': '8.076923 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.3028846 0.389423 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.1298077 0.1298077 0.3317308 0.2307692 0.2307692 0.2884615 0.403846 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2740385 0.418269 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.4615385 0.4615385 0.2307692 0.2307692 0.2740385 0.418269 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2884615 0.403846 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.1153846 0.1153846 0.3461539 0.2740385 0.2740385 0.2307692 0.418269 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2740385 0.418269 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.4615385 0.4615385 0.2596154 0.2596154 0.3173077 0.346154 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.4615385 0.4615385 0.1442308 0.1442308 0.4182692 0.360577 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.4615385 0.4615385 0.1586538 0.1586538 0.3894231 0.375 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.2307692 0.2307692 0.1586538 0.1586538 0.4615385 0.302885 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.1298077 0.1298077 0.3317308 0.2307692 0.2307692 0.4615385 0.4615385 0.4615385 0.4615385 0.1442308 0.1442308 0.3461539 0.432692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.5480769 0.5480769 0.375 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.5480769 0.5480769 0.375 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.1153846 0.1153846 0.3461539 0.2740385 0.2740385 0.4182692 0.375 0.317308 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.375 0.375 0.317308 0.2307692 0.2307692 0.4615385 0.4615385 0.2740385 0.2740385 0.1875 0.2307692 0.2307692 0.230769 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.4615385 0.4615385 0.1442308 0.1442308 0.4326923 0.346154 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.5913461 0.5913461 0.331731 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.5913461 0.5913461 0.331731 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.1298077 0.3317308 0.2884615 0.2884615 0.4038461 0.3028846 0.389423 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.3894231 0.3894231 0.302885 0.2307692 0.2307692 0.4615385 0.4615385 0.2740385 0.2740385 0.1875 0.1730769 0.1730769 0.288462 0.2307692 0.2307692 0.4615385 0.4615385 0.2307692 0.2307692 0.4615385 0.4615385 0.1298077 0.1298077 0.3317308 0.2163462 0.2163462 0.4759615 0.3894231 1',
        'is_slur_seq': '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0',
        'ph_dur': '7.911923 0.165 0.12718 0.103589 0.185769 0.045 0.155769 0.075 0.13782 0.092949 0.185769 0.045 0.416538 0.045 0.124423 0.106346 0.416538 0.045 0.185769 0.045 0.155768 0.075001 0.230769 0.302885 0.314423 0.075 0.107052 0.123717 0.185769 0.045 0.167052 0.063717 0.17077 0.059999 0.170769 0.06 0.401666 0.059873 0.185769 0.045 0.290193 0.171346 0.129808 0.188012 0.143719 0.230769 0.288462 0.343847 0.059999 0.131732 0.099037 0.185769 0.045 0.185769 0.045 0.1475 0.083269 0.185769 0.045 0.371538 0.09 0.142179 0.088591 0.311539 0.15 0.155768 0.075001 0.177501 0.053268 0.230769 0.274038 0.393268 0.025002 0.187821 0.042948 0.185769 0.045 0.185769 0.045 0.150062 0.080708 0.185384 0.045385 0.356537 0.105001 0.119102 0.111668 0.311537 0.150002 0.356539 0.105 0.230769 0.274038 0.373269 0.045 0.102946 0.127823 0.185769 0.045 0.185769 0.045 0.185771 0.044998 0.108523 0.122246 0.401537 0.060001 0.111154 0.119616 0.416538 0.045 0.172755 0.058014 0.15577 0.074999 0.230769 0.288462 0.358846 0.045 0.111602 0.119167 0.15577 0.074999 0.162692 0.068077 0.15577 0.074999 0.131024 0.099745 0.311539 0.15 0.185769 0.045 0.317692 0.143847 0.115385 0.196152 0.150002 0.274038 0.230769 0.373271 0.044998 0.124934 0.105835 0.118781 0.111988 0.185771 0.044998 0.155768 0.075001 0.127177 0.103592 0.41654 0.044998 0.127563 0.103207 0.41654 0.044998 0.129546 0.101223 0.14532 0.085449 0.230769 0.274038 0.393268 0.025002 0.17686 0.053909 0.170768 0.060001 0.185771 0.044998 0.185767 0.045002 0.114741 0.116028 0.356539 0.105 0.150062 0.080708 0.301085 0.160454 0.290259 0.17128 0.259615 0.317308 0.300961 0.045193 0.15673 0.074039 0.203528 0.027241 0.197818 0.032951 0.169616 0.061153 0.151668 0.079102 0.41654 0.044998 0.132499 0.09827 0.356535 0.105003 0.385771 0.075768 0.144231 0.418269 0.317951 0.042625 0.103847 0.126923 0.154811 0.075958 0.185767 0.045002 0.170772 0.059998 0.127372 0.103397 0.416536 0.045002 0.139617 0.091152 0.386538 0.075001 0.312758 0.148781 0.158654 0.389423 0.314999 0.060001 0.116088 0.114681 0.185767 0.045002 0.155768 0.075001 0.185771 0.044998 0.087241 0.143528 0.34532 0.116219 0.182818 0.047951 0.356539 0.105 0.155768 0.075001 0.154998 0.075771 0.158654 0.461538 0.257883 0.045002 0.128524 0.102245 0.202945 0.027824 0.097816 0.132954 0.155772 0.074997 0.168716 0.062054 0.129808 0.271729 0.060001 0.164615 0.066154 0.323973 0.137566 0.308973 0.152565 0.144231 0.346154 0.372691 0.060001 0.185771 0.044998 0.185767 0.045002 0.185771 0.044998 0.109038 0.121731 0.548077 0.240002 0.134998 0.185767 0.045002 0.185771 0.044998 0.185767 0.045002 0.109233 0.121536 0.548077 0.330002 0.044998 0.116408 0.114361 0.185767 0.045002 0.170577 0.060192 0.185771 0.044998 0.111991 0.118778 0.41654 0.044998 0.185767 0.045002 0.386538 0.075001 0.115385 0.245194 0.10096 0.274038 0.418269 0.375 0.281472 0.035835 0.170768 0.060001 0.15301 0.077759 0.10942 0.121349 0.386538 0.075001 0.127177 0.103592 0.313074 0.148464 0.375 0.272302 0.045006 0.101024 0.129745 0.356543 0.104996 0.274038 0.10878 0.07872 0.230769 0.140773 0.089996 0.129741 0.101028 0.41654 0.044998 0.114749 0.11602 0.326536 0.135002 0.385962 0.075577 0.144231 0.432692 0.301156 0.044998 0.1114 0.11937 0.185771 0.044998 0.185771 0.044998 0.119479 0.11129 0.591346 0.196736 0.134995 0.185771 0.044998 0.155765 0.075005 0.185771 0.044998 0.11967 0.111099 0.591346 0.226735 0.104996 0.144809 0.08596 0.155765 0.075005 0.155765 0.075005 0.170772 0.059998 0.151668 0.079102 0.41654 0.044998 0.185771 0.044998 0.401533 0.189813 0.211423 0.120308 0.288462 0.403846 0.302885 0.344417 0.045006 0.170764 0.060005 0.132129 0.09864 0.118671 0.112099 0.401533 0.060005 0.146937 0.083832 0.323286 0.138252 0.389423 0.197889 0.104996 0.155772 0.074997 0.358343 0.103195 0.274038 0.142502 0.044998 0.173077 0.153459 0.135002 0.124492 0.106277 0.341543 0.119995 0.121677 0.109093 0.29225 0.169289 0.129808 0.173413 0.158318 0.216346 0.475962 0.389423 0.05',
        'input_type': 'phoneme' # Manual phone duration mode
    }  # input like Opencpop dataset.
    DiffSingerE2EInfer.example_run(inp3)


# python inference/ds_e2e.py --config configs/midi/e2e/opencpop/ds100_adj_rel.yaml --exp_name 0228_opencpop_ds100_rel
