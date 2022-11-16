from utils.hparams import hparams


_has_cache = False
_g2p_dictionary = {
    'AP': ['AP'],
    'SP': ['SP']
}
_phoneme_list: list


def _build_dict_and_list():
    global _g2p_dictionary, _phoneme_list
    _set = set()
    with open(hparams['g2p_dictionary'], 'r', encoding='utf8') as _df:
        _lines = _df.readlines()
    for _line in _lines:
        _pinyin, _ph_str = _line.strip().split('\t')
        _g2p_dictionary[_pinyin] = _ph_str.split()
    for _list in _g2p_dictionary.values():
        [_set.add(ph) for ph in _list]
    _phoneme_list = sorted(list(_set))
    print('| build phone set:', _phoneme_list)


def build_g2p_dictionary() -> dict:
    global _has_cache
    if not _has_cache:
        _build_dict_and_list()
        _has_cache = True
    return _g2p_dictionary


def build_phoneme_list() -> list:
    global _has_cache
    if not _has_cache:
        _build_dict_and_list()
        _has_cache = True
    return _phoneme_list


def opencpop_old_to_strict(phonemes: list, slurs: list) -> list:
    assert len(phonemes) == len(slurs), 'Length of phonemes mismatches length of slurs!'
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
    with open('../dictionaries/transcriptions-revised.txt', 'r', encoding='utf8') as f:
        _utterances = f.readlines()
    utterances: list = [u.strip().split('|') for u in _utterances]
    for u in utterances:
        u[2] = ' '.join(opencpop_old_to_strict(u[2].split(), u[6].split()))
    with open('../dictionaries/transcriptions-strict.txt', 'w', encoding='utf-8') as f:
        for u in utterances:
            print('|'.join(u), file=f)
