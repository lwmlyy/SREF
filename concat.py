import logging
import argparse
import numpy as np
from datetime import datetime
from orderedset import OrderedSet


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def normalize(vec_as_list):
    vec = np.array(vec_as_list)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def concat(v2_path, v3_path, out_path=None, norm=True):
    txt2_vecs = {}
    if type(v2_path) is str:
        with open(v2_path) as txt2_f:
            for line in txt2_f:
                info = line.split()
                label, vec_str = info[0], info[1:]
                vec = [float(v) for v in vec_str]
                if norm:
                    vec = normalize(vec)
                txt2_vecs[label] = vec
    else:
        for label, vec_str in v2_path.items():
            vec = [float(v) for v in vec_str]
            if norm:
                vec = normalize(vec)
            txt2_vecs[label] = vec

    logging.info('Loading %s ...' % v3_path)  # i.e. BERT tokens
    txt3_vecs = {}
    if v3_path is not None:
        import pickle
        sense_dict = pickle.load(open(v3_path, 'rb'))
        for label, vec_str in sense_dict.items():
            vec = [float(v) for v in vec_str]
            if norm:
                vec = normalize(vec)
            txt3_vecs[label] = vec

    logging.info('Combining vecs (concat) ...')
    txt1_labels = OrderedSet(txt2_vecs.keys())  # first sets the order
    for label1 in txt1_labels:
        v2 = txt2_vecs[label1]

        if v3_path is not None:
            try:
                v3 = txt3_vecs[label1]  # takes from txt2 if missing
            except:
                v3 = txt2_vecs[label1]
            txt2_vecs[label1] = v2 + v3  # concatenation, not sum
        else:
            v1 = [0] * 1024
            txt2_vecs[label1] = v1 + v2  # concatenation, not sum

    if out_path:
        logging.info('Writing %s ...' % out_path)
        with open(out_path, 'w') as merged_f:
            for label in txt1_labels:
                vec = txt2_vecs[label]
                vec_str = [str(round(v, 6)) for v in vec]
                merged_f.write('%s %s\n' % (label, ' '.join(vec_str)))
    else:
        return txt2_vecs
    logging.info('Done')
