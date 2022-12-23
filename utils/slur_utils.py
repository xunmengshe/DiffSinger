def merge_slurs(param):
    ph_seq = param['ph_seq'].split()
    note_seq = param['note_seq'].split()
    note_dur_seq = param['note_dur_seq'].split()
    is_slur_seq = [int(s) for s in param['is_slur_seq'].split()]
    ph_dur = [float(d) for d in param['ph_dur'].split()]
    i = 0
    while i < len(ph_seq):
        if is_slur_seq[i]:
            ph_dur[i - 1] += ph_dur[i]
            ph_seq.pop(i)
            note_seq.pop(i)
            note_dur_seq.pop(i)
            is_slur_seq.pop(i)
            ph_dur.pop(i)
        else:
            i += 1
    param['ph_seq'] = ' '.join(ph_seq)
    param['note_seq'] = ' '.join(note_seq)
    param['note_dur_seq'] = ' '.join(note_dur_seq)
    param['is_slur_seq'] = ' '.join([str(s) for s in is_slur_seq])
    param['ph_dur'] = ' '.join([str(d) for d in ph_dur])
