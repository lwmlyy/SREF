import os
import logging
import argparse
from time import time
from datetime import datetime
from functools import lru_cache
from collections import defaultdict
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
from nltk.corpus import wordnet as wn

from bert_as_service import bert_embed
from vectorspace import SensesVSM
from vectorspace import get_sk_pos

from concat import *
from synset_expand import *
import pickle

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def load_wsd_fw_set(wsd_fw_set_path):
    """Parse XML of split set and return list of instances (dict)."""
    eval_instances = []
    tree = ET.parse(wsd_fw_set_path)
    for text in tree.getroot():
        for sent_idx, sentence in enumerate(text):
            inst = {'tokens': [], 'tokens_mw': [], 'lemmas': [], 'senses': [], 'pos': []}
            for e in sentence:
                inst['tokens_mw'].append(e.text)
                inst['lemmas'].append(e.get('lemma'))
                inst['senses'].append(e.get('id'))
                inst['pos'].append(e.get('pos'))

            inst['tokens'] = sum([t.split() for t in inst['tokens_mw']], [])

            # handling multi-word expressions, mapping allows matching tokens with mw features
            idx_map_abs = []
            idx_map_rel = [(i, list(range(len(t.split()))))
                            for i, t in enumerate(inst['tokens_mw'])]
            token_counter = 0
            for idx_group, idx_tokens in idx_map_rel:  # converting relative token positions to absolute
                idx_tokens = [i+token_counter for i in idx_tokens]
                token_counter += len(idx_tokens)
                idx_map_abs.append([idx_group, idx_tokens])

            inst['tokenized_sentence'] = ' '.join(inst['tokens'])
            inst['idx_map_abs'] = idx_map_abs
            inst['idx'] = sent_idx

            eval_instances.append(inst)

    return eval_instances


@lru_cache()
def wn_sensekey2synset(sensekey):
    """Convert sensekey to synset."""
    lemma = sensekey.split('%')[0]
    for synset in wn.synsets(lemma):
        for lemma in synset.lemmas():
            if lemma.key() == sensekey:
                return synset
    return None


def get_id2sks(wsd_eval_keys):
    """Maps ids of split set to sensekeys, just for in-code evaluation."""
    id2sks = {}
    with open(wsd_eval_keys) as keys_f:
        for line in keys_f:
            id_ = line.split()[0]
            keys = line.split()[1:]
            id2sks[id_] = keys
    return id2sks


def run_scorer(wsd_fw_path, test_set, results_path):
    """Runs the official java-based scorer of the WSD Evaluation Framework."""
    cmd = 'cd %s && java Scorer %s %s' % (wsd_fw_path + 'Evaluation_Datasets/',
                                          '%s/%s.gold.key.txt' % (test_set, test_set),
                                          '../../../../' + results_path)
    print(cmd)
    os.system(cmd)


def chunks(l, n):
    """Yield successive n-sized chunks from given list."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def str_scores(scores, n=3, r=5):
    """Convert scores list to a more readable string."""
    return str([(l, round(s, r)) for l, s in scores[:n]])


def sec_wsd(matches):
    from extend import wn_all_lexnames_groups
    lexname_groups = wn_all_lexnames_groups()
    preds = [sk for sk, sim in matches if sim > args.thresh][:]
    preds_sim = [sim for sk, sim in matches if sim > args.thresh][:]
    norm_predsim = np.exp(preds_sim) / np.sum(np.exp(preds_sim))
    name = locals()
    if len(preds) != 1:
        pos2type = {'ADJ': 'as', 'ADV': 'r', 'NOUN': 'n', 'VERB': 'v'}
        synset_list = retrieve_sense(curr_lemma, pos2type[curr_postag])
        keys = [k[0] for k in matches][:2]
        try:
            synsets = {wn.lemma_from_key(j).synset(): i for i, j in enumerate(keys)}
        except:
            synsets = {
                [wn.synset(k) for k in synset_list if j in [l.key() for l in wn.synset(k).lemmas()]][0]: i for
                i, j in enumerate(keys)}
        strategy = 'r_sy+relations'
        # print([i.lexname() for i in synsets])
        all_related = Counter()
        for potential_synset in synsets.keys():
            name[potential_synset.name()] = set(gloss_extend(potential_synset.name(), strategy))
            all_related.update(list(name[potential_synset.name()]))

        for synset, count in all_related.items():
            if count == 1:
                continue
            for potential_synset in synsets.keys():
                while synset in name[potential_synset.name()]:
                    name[potential_synset.name()].remove(synset)

        for synset_index, potential_synset in enumerate(synsets.keys()):
            lexname = potential_synset.lexname()
            name['sim_%s' % potential_synset.name()] = dict()

            if len(set([i.lexname() for i in synsets])) > 1:
                combine_list = list(name[potential_synset.name()]) + lexname_groups[lexname]
            else:
                combine_list = list(name[potential_synset.name()])
            for synset in combine_list:
                if synset in synsets.keys() and curr_postag not in ['ADJ', 'ADV']:
                    continue
                sim = np.dot(curr_vector, senses_vsm.get_vec(synset.lemmas()[0].key()))
                name['sim_%s' % potential_synset.name()][synset] = (
                    sim, 'relation' if synset in name[potential_synset.name()] else 'lexname')

        key_score = {keys[j]: preds_sim[j] + np.sum(
            sorted([syn[0] for syn in name['sim_%s' % i.name()].values()], reverse=True)[:1]) for i, j in
                     synsets.items()}

        final_key = [sorted(key_score.items(), key=lambda x: x[1], reverse=True)[0][0]]

    else:
        final_key = preds
    
    return final_key


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Nearest Neighbors WSD Evaluation.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-lmms_path', default='data/vectors/lmms.txt', required=False,
                        help='Path to LMMS vector')
    parser.add_argument('-wsd_fw_path', help='Path to WSD Evaluation Framework', required=False,
                        default='data/wsd_eval/WSD_Evaluation_Framework/')
    parser.add_argument('-emb_strategy', type=str, default='aug_gloss+r_sy',
                        choices=['aug_gloss+r_sy+examples', 'aug_gloss+r_sy+examples+lmms'],
                        help='different components to learn the basic sense embeddings', required=False)
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size (BERT)', required=False)
    parser.add_argument('-merge_strategy', type=str, default='mean', help='WordPiece Reconstruction Strategy', required=False)
    parser.add_argument('-ignore_lemma', dest='use_lemma', action='store_false', help='Ignore lemma features', required=False)
    parser.add_argument('-ignore_pos', dest='use_pos', action='store_false', help='Ignore POS features', required=False)
    parser.add_argument('-sec_wsd', default=False, help='whether to implement second wsd', required=False)
    parser.add_argument('-thresh', type=float, default=-1, help='Similarity threshold', required=False)
    parser.add_argument('-k', type=int, default=1, help='Number of Neighbors to accept', required=False)
    parser.set_defaults(use_lemma=True)
    parser.set_defaults(use_pos=True)
    parser.set_defaults(debug=True)
    args = parser.parse_args()

    """
    Load sense embeddings for evaluation.
    Check the dimensions of the sense embeddings to guess that they are composed with static embeddings.
    """
    logging.info('Loading SensesVSM ...')
    
    emb_wn = pickle.load(open('data/vectors/emb_glosses_hyper_aug_gloss+r_sy+examples.txt', 'rb'))
    if 'lmms' in args.emb_strategy:
        lmms = pickle.load(open(args.lmms_path, 'rb'))

        for sense_key, sense_vector in emb_wn.items():
                if sense_key in lmms:
                    lmms[sense_key] = np.array(lmms[sense_key])/np.linalg.norm(np.array(lmms[sense_key]))
                    emb_wn[sense_key] = sense_vector + lmms[sense_key].tolist()
                else:
                    emb_wn[sense_key] = emb_wn[sense_key] + emb_wn[sense_key]
    logging.info('Loaded SensesVSM')
    senses_vsm = SensesVSM(emb_wn, normalize=True)

    """
    Initialize various counters for calculating supplementary metrics for ALL dataset.
    """
    num_all, num_correct = 0, 0
    pos_correct, pos_all = np.array([0]*4), np.array([0]*4)
    mfs_correct, mfs_all = 0, 0
    lfs_correct, lfs_all = 0, 0
    pos_position = ['NOUN', 'VERB', 'ADJ', 'ADV']
    for data_index, test_set in enumerate(['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015']):
        """
        Initialize various counters for calculating supplementary metrics for each test set.
        """
        n_instances, n_correct, n_unk_lemmas = 0, 0, 0
        correct_idxs = []
        num_options = []
        failed_by_pos = defaultdict(list)

        pos_confusion = {}
        for pos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            pos_confusion[pos] = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0}

        """
        Load evaluation instances and gold labels.
        Gold labels (sensekeys) only used for reporting accuracy during evaluation.
        """
        wsd_fw_set_path = args.wsd_fw_path + 'Evaluation_Datasets/%s/%s.data.xml' % (test_set, test_set)
        wsd_fw_gold_path = args.wsd_fw_path + 'Evaluation_Datasets/%s/%s.gold.key.txt' % (test_set, test_set)
        id2senses = get_id2sks(wsd_fw_gold_path)
        eval_instances = load_wsd_fw_set(wsd_fw_set_path)

        """
        Iterate over evaluation instances and write predictions in WSD_Evaluation_Framework's format.
        File with predictions is processed by the official scorer after iterating over all instances.
        """
        results_path = 'data/results/%s.%s.%s.key' % (args.emb_strategy, test_set, args.merge_strategy)
        with open(results_path, 'w') as results_f:
            for batch_idx, batch in enumerate(chunks(eval_instances, args.batch_size)):
                batch_sents = [sent_info['tokenized_sentence'] for sent_info in batch]

                # process contextual embeddings in sentences batches of size args.batch_size
                batch_bert = bert_embed(batch_sents, merge_strategy=args.merge_strategy)

                for sent_info, sent_bert in zip(batch, batch_bert):
                    idx_map_abs = sent_info['idx_map_abs']

                    for mw_idx, tok_idxs in idx_map_abs:
                        curr_sense = sent_info['senses'][mw_idx]

                        if curr_sense is None:
                            continue

                        curr_lemma = sent_info['lemmas'][mw_idx]

                        if args.use_lemma and curr_lemma not in senses_vsm.known_lemmas:
                            continue  # skips hurt performance in official scorer

                        curr_postag = sent_info['pos'][mw_idx]
                        curr_tokens = [sent_info['tokens'][i] for i in tok_idxs]
                        curr_vector = np.array([sent_bert[i][1] for i in tok_idxs]).mean(axis=0)
                        curr_vector = curr_vector / np.linalg.norm(curr_vector)

                        """
                        Compose test-time embedding for matching with sense embeddings in SensesVSM.
                        Test-time embedding corresponds to stack of contextual and (possibly) static embeddings.
                        Stacking composition performed according to dimensionality of sense embeddings.
                        """
                        if senses_vsm.ndims == 1024:
                            curr_vector = curr_vector

                        # duplicating contextual feature for cos similarity against features from
                        # sense annotations and glosses that belong to the same NLM
                        elif senses_vsm.ndims == 1024+1024:
                            curr_vector = np.hstack((curr_vector, curr_vector))

                        curr_vector = curr_vector / np.linalg.norm(curr_vector)

                        """
                        Matches test-time embedding against sense embeddings in SensesVSM.
                        use_lemma and use_pos flags condition filtering of candidate senses.
                        Matching is actually cosine similarity (most similar), or 1-NN.
                        """
                        matches = []
                        if args.use_lemma and curr_lemma not in senses_vsm.known_lemmas:
                            n_unk_lemmas += 1

                        elif args.use_lemma and args.use_pos:  # the usual for WSD
                            matches = senses_vsm.match_senses(curr_vector, curr_lemma, curr_postag, topn=None)

                        elif args.use_lemma:
                            matches = senses_vsm.match_senses(curr_vector, curr_lemma, None, topn=None)

                        elif args.use_pos:
                            matches = senses_vsm.match_senses(curr_vector, None, curr_postag, topn=None)

                        else:  # corresponds to Uninformed Sense Matching (USM)
                            matches = senses_vsm.match_senses(curr_vector, None, None, topn=None)

                        num_options.append(len(matches))

                        # predictions can be further filtered by similarity threshold or number of accepted neighbors
                        # if specified in CLI parameters
                        preds = [sk for sk, sim in matches if sim > args.thresh][:args.k]
                        if args.sec_wsd:
                            preds = sec_wsd(matches)[:1]
                        if len(preds) > 0:
                            results_f.write('%s %s\n' % (curr_sense, preds[0]))

                        """
                        Processing additional performance metrics.
                        """

                        # check if our prediction(s) was correct, register POS of mistakes
                        n_instances += 1
                        wsd_correct = False
                        gold_sensekeys = id2senses[curr_sense]
                        pos_dict = {'NOUN': 'n', 'ADJ': 'a', 'ADV': 'r', 'VERB': 'v'}
                        wn1st = [i.key() for i in wn.synsets(curr_lemma, pos_dict[curr_postag])[0].lemmas()]
                        if len(set(preds).intersection(set(gold_sensekeys))) > 0:
                            n_correct += 1
                            wsd_correct = True
                            if len(preds) > 0:
                                failed_by_pos[curr_postag].append((preds[0], gold_sensekeys))
                            else:
                                failed_by_pos[curr_postag].append((None, gold_sensekeys))

                        # register if our prediction belonged to a different POS than gold
                        if len(preds) > 0:
                            pred_sk_pos = get_sk_pos(preds[0])
                            gold_sk_pos = get_sk_pos(gold_sensekeys[0])
                            pos_confusion[gold_sk_pos][pred_sk_pos] += 1

                        # register how far the correct prediction was from the top of our matches
                        correct_idx = None
                        for idx, (matched_sensekey, matched_score) in enumerate(matches):
                            if matched_sensekey in gold_sensekeys:
                                correct_idx = idx
                                correct_idxs.append(idx)
                                break

        logging.info('Running official scorer ...')
        run_scorer(args.wsd_fw_path, test_set, results_path)
        num_all += n_instances
        num_correct += n_correct
        pos_all += np.array([sum(pos_confusion[i].values()) for i in pos_position])
        pos_correct += np.array([len(failed_by_pos[i]) for i in pos_position])
    print('F-all %f' % (num_correct/num_all))
    print(pos_position, pos_all.tolist(), pos_correct.tolist(), (pos_correct/pos_all).tolist())
