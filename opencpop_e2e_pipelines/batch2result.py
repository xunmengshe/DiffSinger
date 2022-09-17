'''
    batch -> insert1 -> module1 -> insert2 -> module2 -> insert3 -> module3 -> insert_final -> result
'''
from pipelines import batch2result

class Batch2Result(batch2result.Batch2Result):
    '''
        batch -> insert1 -> module1 -> insert2 -> module2 -> insert3 -> module3 -> insert_final -> result
    '''

    @staticmethod
    def insert1(pitch_midi, midi_dur, is_slur, midi_embed, midi_dur_layer, is_slur_embed):
        '''
            add embeddings for midi, midi_dur, is_slur
        '''
        midi_embedding = midi_embed(pitch_midi)
        midi_dur_embedding, slur_embedding = 0, 0
        if midi_dur is not None:
            midi_dur_embedding = midi_dur_layer(midi_dur[:, :, None])  # [B, T, 1] -> [B, T, H]
        if is_slur is not None:
            slur_embedding = is_slur_embed(is_slur)
        return midi_embedding, midi_dur_embedding, slur_embedding

    @staticmethod
    def module1(fs2_encoder, txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding):
        '''
            fastspeech2 encoder
        '''
        return fs2_encoder(txt_tokens, midi_embedding, midi_dur_embedding, slur_embedding)
    
    @staticmethod
    def insert2():
        raise NotImplementedError

    @staticmethod
    def module2():
        raise NotImplementedError
    
    @staticmethod
    def insert3():
        raise NotImplementedError

    @staticmethod
    def module3():
        raise NotImplementedError
    
    @staticmethod
    def insert_final():
        raise NotImplementedError
