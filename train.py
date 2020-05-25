import logging
import argparse
from time import time

import lxml.etree
import numpy as np
import re
from nltk.corpus import wordnet as wn
from bert_as_service import bert_embed
from bert_as_service import tokenizer as bert_tokenizer
import json
from tqdm import tqdm
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def get_sense_mapping(eval_path):
    sensekey_mapping = {}
    with open(eval_path) as keys_f:
        for line in keys_f:
            id_ = line.split()[0]
            keys = line.split()[1:]
            sensekey_mapping[id_] = keys
    return sensekey_mapping


def read_xml_sents(xml_path):
    with open(xml_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('<sentence '):
                sent_elems = [line]
            elif line.startswith('<wf ') or line.startswith('<instance '):
                sent_elems.append(line)
            elif line.startswith('</sentence>'):
                sent_elems.append(line)
                yield lxml.etree.fromstring(''.join(sent_elems))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_adj_keys():
    key_list = []
    for synset in wn.all_synsets('a'):
        for lemma in synset.lemmas():
            key_list.extend([lemma.key()])
    return key_list


def train(vecs_path, dataset, merge_strategy='mean', max_instances=float('inf')):
    batch_all = []
    sense_vecs = {}
    if 'semcor' not in dataset:
        new_dataset = dataset
        print('loading %s' % new_dataset)
        extra_path = 'path to your training data' + new_dataset
        wngt_corpus = open(extra_path, 'r').read()
        wsd_bs = BeautifulSoup(wngt_corpus, 'xml')
        text_all = wsd_bs.find_all('sentence')
        type2pos = {'j': 'ADJ', 'n': 'NOUN', 'r': 'ADV', 'v': 'VERB'}
        adj_keys = get_adj_keys()
        for sent in tqdm(text_all):
            entry = {f: [] for f in ['token', 'token_mw', 'lemma', 'senses', 'pos', 'id']}
            for word in sent.find_all('word'):
                lemma = word['lemma'] if 'lemma' in word.attrs else word['surface_form'].replace('_', ' ')
                pos = type2pos[word['pos'][0].lower()] if word['pos'][0].lower() in type2pos else word['pos']
                label = word['wn30_key'].split(';') if 'wn30_key' in word.attrs else None
                if label and '%3:' in ''.join(label):
                    for index, key in enumerate(label):
                        if key not in adj_keys and '%3:' in key:
                            pos_string = key.split('%')[1][0]
                            replace_string = '35'.replace(key.split('%')[1][0], '')
                            label[index] = key.replace('%' + pos_string + ':', '%' + replace_string + ':')
                entry['lemma'].append(lemma)
                entry['pos'].append(pos)
                entry['token_mw'].append(word['surface_form'].replace('_', ' '))
                entry['senses'].append(label)
            entry['token'] = sum([t.split() for t in entry['token_mw']], [])
            entry['sentence'] = ' '.join([t for t in entry['token_mw']])
            if any(entry['senses']):
                batch_all.append(entry)

    else:
        train_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.data.xml'
        eval_path = args.wsd_fw_path + 'Training_Corpora/SemCor/semcor.gold.key.txt'
        print('loading %s' % dataset)
        sense_mapping = get_sense_mapping(eval_path)
        for sent_idx, sent_et in enumerate(read_xml_sents(train_path)):
            entry = {f: [] for f in ['token', 'token_mw', 'lemma', 'senses', 'pos', 'id']}
            for ch in sent_et.getchildren():
                for k, v in ch.items():
                    entry[k].append(v)
                entry['token_mw'].append(ch.text)
                if 'id' in ch.attrib.keys():
                    entry['senses'].append(sense_mapping[ch.attrib['id']])
                else:
                    entry['senses'].append(None)
            entry['token'] = sum([t.split() for t in entry['token_mw']], [])
            entry['sentence'] = ' '.join([t for t in entry['token_mw']])
            if entry['sentence']:
                batch_all.append(entry)

    if args.cut_train:
        batch_all = batch_all[:int(len(batch_all)*(args.portion/10))]

    print('sents number: %s, batch number: %s' % (str(len(batch_all)), str(len(batch_all) / args.batch_size)))
    for batch_idx, batch in enumerate(tqdm(chunks(batch_all, args.batch_size))):
        batch_sents = [e['sentence'] for e in batch]
        batch_bert = bert_embed(batch_sents, merge_strategy=merge_strategy)

        for sent_info, sent_bert in zip(batch, batch_bert):
            # handling multi-word expressions, mapping allows matching tokens with mw features
            idx_map_abs = []
            idx_map_rel = [(i, list(range(len(t.split()))))
                           for i, t in enumerate(sent_info['token_mw'])]
            token_counter = 0
            for idx_group, idx_tokens in idx_map_rel:  # converting relative token positions to absolute
                idx_tokens = [i + token_counter for i in idx_tokens]
                token_counter += len(idx_tokens)
                idx_map_abs.append([idx_group, idx_tokens])

            for mw_idx, tok_idxs in idx_map_abs:
                if sent_info['senses'][mw_idx] is None:
                    continue
                vec = np.array([sent_bert[i][1] for i in tok_idxs], dtype=np.float32).mean(axis=0)
                for sense in sent_info['senses'][mw_idx]:
                    try:
                        if sense_vecs[sense]['vecs_num'] < max_instances:
                            sense_vecs[sense]['vecs_sum'] += vec
                            sense_vecs[sense]['vecs_num'] += 1
                    except KeyError:
                        sense_vecs[sense] = {'vecs_sum': vec, 'vecs_num': 1}

    logging.info('Writing Sense Vectors ...')

    import pickle
    sense_dict = dict()
    for sense, vecs_info in sense_vecs.items():
        vec = vecs_info['vecs_sum'] / vecs_info['vecs_num']
        sense_dict[sense] = vec
    pickle.dump(sense_dict, open(args.out_path % args.portion, 'wb'), -1)
    logging.info('Written %s' % vecs_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Initial Sense Embeddings.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-wsd_fw_path', help='Path to WSD Evaluation Framework', required=False,
                        default='data/wsd_eval/WSD_Evaluation_Framework/')
    parser.add_argument('-dataset', default='semcor', help='Name of dataset', required=False,
                        choices=['semcor', 'senseval2-LS', 'senseval3-LS'])
    parser.add_argument('-batch_size', type=int, default=64, help='Batch size (BERT)', required=False)
    parser.add_argument('-max_seq_len', type=int, default=64, help='Maximum sequence length (BERT)', required=False)
    parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False,
                        choices=['mean', 'first', 'sum'])
    parser.add_argument('-cut_train', type=bool, default=False, help='whether to cut training data', required=False)
    parser.add_argument('-portion', type=int, default=1, help='portion of SemCor used for training', required=False)
    parser.add_argument('-max_instances', type=float, default=float('inf'),
                        help='Maximum number of examples for each sense', required=False)
    parser.add_argument('-out_path', help='Path to resulting vector set', required=False, default='your output path')
    args = parser.parse_args()
    train(args.out_path, args.dataset, args.merge_strategy, args.portion)
