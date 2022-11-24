"""

    item: one piece of data
    item_name: data id
    wavfn: wave file path
    txt: lyrics
    ph: phoneme
    tgfn: text grid file path (unused)
    spk: dataset name
    wdb: word boundary
    ph_durs: phoneme durations
    midi: pitch as midi notes
    midi_dur: midi duration
    is_slur: keep singing upon note changes
"""

import logging
from copy import deepcopy

from data_gen.midisinging import MidiSingingBinarizer
from utils.hparams import hparams


class OpencpopBinarizer(MidiSingingBinarizer):
    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)
        test_item_names = [x for x in item_names if any([x.startswith(ts) for ts in hparams['test_prefixes']])]
        train_item_names = [x for x in item_names if x not in set(test_item_names)]
        logging.info("train {}".format(len(train_item_names)))
        logging.info("test {}".format(len(test_item_names)))
        return train_item_names, test_item_names

    def load_meta_data(self, processed_data_dir, ds_id):
        from preprocessing.opencpop import File2Batch
        self.items = File2Batch.file2temporary_dict()