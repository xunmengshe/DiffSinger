import argparse
import json


_has_cache = False
_g2p_dictionary = {
    'AP': ['AP'],
    'SP': ['SP']
}
_phoneme_list: list


def _build_dict_and_list():
    from utils.hparams import hparams
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
    print('| load phone set:', _phoneme_list)


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


def opencpop_ds_old_to_strict(ds_params):
    ds_params['ph_seq'] = ' '.join(
        opencpop_old_to_strict(ds_params['ph_seq'].split(), ds_params['is_slur_seq'].split()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Migrate ds file from old opencpop pinyin dictionary to new strict pinyin dictionary.')
    parser.add_argument('input', type=str, help='Path to the input file')
    parser.add_argument('output', type=str, help='Path to the output file')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf8') as f:
        params = json.load(f)
    if isinstance(params, list):
        [opencpop_ds_old_to_strict(p) for p in params]
    else:
        opencpop_ds_old_to_strict(params)

    with open(args.output, 'w', encoding='utf8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
