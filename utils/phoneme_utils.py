import os.path

from utils.hparams import hparams


ph_dict = {}
_dictionary = 'opencpop-strict.txt' if hparams.get('use_strict_yunmu') else 'opencpop.txt'
with open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'phoneme',
        _dictionary
), 'r') as _df:
    _lines = _df.readlines()
for _line in _lines:
    _pinyin, _ph_str = _line.strip().split('|')
    ph_dict[_pinyin] = _ph_str.split()


def pinyin_to_phoneme(pinyin: str) -> list:
    return ph_dict[pinyin]


def migrate_from_old(phonemes: list, slurs: list) -> list:
    assert len(phonemes) == len(slurs), 'Length of phonemes mismatches length of slur!'
    new_phonemes = [p for p in phonemes]
    i = 0
    while i < len(phonemes):
        if phonemes[i] == 'i' and i > 0:
            rep = None
            if phonemes[i - 1] in ['zh', 'ch', 'sh', 'r']:
                rep = 'ir'
            elif phonemes[i - 1] in ['z', 'c', 's']:
                rep = 'i0'
            if rep is not None:
                new_phonemes[i] = rep
                i += 1
                while i < len(phonemes) and slurs[i] == '1':
                    new_phonemes[i] = rep
                    i += 1
            else:
                i += 1
        elif phonemes[i] == 'e' and i > 0 and phonemes[i - 1] == 'y':
            new_phonemes[i] = 'E'
            i += 1
            while i < len(phonemes) and slurs[i] == '1':
                new_phonemes[i] = 'E'
                i += 1
        elif phonemes[i] == 'an' and i > 0 and phonemes[i - 1] == 'y':
            new_phonemes[i] = 'En'
            i += 1
            while i < len(phonemes) and slurs[i] == '1':
                new_phonemes[i] = 'En'
                i += 1
        else:
            i += 1
    return new_phonemes


if __name__ == '__main__':
    with open('../phoneme/transcriptions-revised.txt', 'r', encoding='utf8') as f:
        _utterances = f.readlines()
    utterances: list = [u.strip().split('|') for u in _utterances]
    for u in utterances:
        u[2] = ' '.join(migrate_from_old(u[2].split(), u[6].split()))
    with open('../phoneme/transcriptions-strict.txt', 'w', encoding='utf-8') as f:
        for u in utterances:
            print('|'.join(u), file=f)
