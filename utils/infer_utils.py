import json

import numpy as np

head_list = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result


def trans_f0_seq(feature_pit, transform):
    feature_pit = feature_pit * 2 ** (transform / 12)
    return round(feature_pit, 1)


def move_key(raw_data, mv_key):
    head = raw_data[:-1]
    body = int(raw_data[-1])
    new_head_index = head_list.index(head) + mv_key
    while new_head_index < 0:
        body -= 1
        new_head_index += 12
    while new_head_index > 11:
        body += 1
        new_head_index -= 12
    result_data = head_list[new_head_index] + str(body)
    return result_data


def trans_key(raw_data, key):
    warning_tag = False
    for i in raw_data:
        note_seq_list = i["note_seq"].split(" ")
        new_note_seq_list = []
        for note_seq in note_seq_list:
            if note_seq != "rest":
                new_note_seq = move_key(note_seq, key)
                new_note_seq_list.append(new_note_seq)
            else:
                new_note_seq_list.append(note_seq)
        i["note_seq"] = " ".join(new_note_seq_list)
        if i["f0_seq"]:
            f0_seq_list = i["f0_seq"].split(" ")
            f0_seq_list = [float(x) for x in f0_seq_list]
            new_f0_seq_list = []
            for f0_seq in f0_seq_list:
                new_f0_seq = trans_f0_seq(f0_seq, key)
                new_f0_seq_list.append(str(new_f0_seq))
            i["f0_seq"] = " ".join(new_f0_seq_list)
        else:
            warning_tag = True
    if warning_tag:
        print("Warning:parts of f0_seq do not exist, please freeze the pitch line in the editor.\r\n")
    return raw_data
