from nltk.corpus import wordnet as wn
import logging
import numpy as np
from collections import Counter
import multiprocessing
from tqdm import tqdm
import pickle
import argparse
from collections import defaultdict

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def normalize(vec_as_list):
    vector = np.array(vec_as_list)
    vector = vector / np.linalg.norm(vector)
    return vector


def retrieve_sense(word, pos=None):
    """
        retrieve sense glosses, sense inventory and sense frequency of word as a dict, list and list respectively
    """
    sense_inventory = [i for i in wn.synsets(word) if i.name().split('.')[-2] in pos]

    name_list, sense_inventory_final = list(), list()
    for sense in sense_inventory:
        lemma_names = [i.name().lower() for i in sense.lemmas()]
        if word.lower() in lemma_names:
            name = sense.name()
            name_list.append(name)
    return name_list


def get_related(names, relation='hypernyms'):
    """
    :param names: the synset list
    :param relation: all the relations
    :return: the extended gloss list with its according synset name
    """
    related_list = []
    for name in names:
        if relation == 'hypernyms':
            wn_relation = wn.synset(name).hypernyms()
        elif relation == 'hyponyms':
            wn_relation = wn.synset(name).hyponyms()
        elif relation == 'part_holonyms':
            wn_relation = wn.synset(name).part_holonyms()
        elif relation == 'part_meronyms':
            wn_relation = wn.synset(name).part_meronyms()
        elif relation == 'member_holonyms':
            wn_relation = wn.synset(name).member_holonyms()
        elif relation == 'member_meronyms':
            wn_relation = wn.synset(name).member_meronyms()
        elif relation == 'entailments':
            wn_relation = wn.synset(name).entailments()
        elif relation == 'attributes':
            wn_relation = wn.synset(name).attributes()
        elif relation == 'also_sees':
            wn_relation = wn.synset(name).also_sees()
        elif relation == 'similar_tos':
            wn_relation = wn.synset(name).similar_tos()
        elif relation == 'causes':
            wn_relation = wn.synset(name).causes()
        elif relation == 'verb_groups':
            wn_relation = wn.synset(name).verb_groups()
        elif relation == 'substance_holonyms':
            wn_relation = wn.synset(name).substance_holonyms()
        elif relation == 'substance_meronyms':
            wn_relation = wn.synset(name).substance_meronyms()
        elif relation == 'usage_domains':
            wn_relation = wn.synset(name).usage_domains()
        elif relation == 'pertainyms':
            wn_relation = [j.synset() for j in sum([i.pertainyms() for i in wn.synset(name).lemmas()], [])]
        elif relation == 'antonyms':
            wn_relation = [j.synset() for j in sum([i.antonyms() for i in wn.synset(name).lemmas()], [])]
        else:
            wn_relation = []
            print('no such relation, process terminated.')
        related_list += wn_relation
    return related_list


def morpho_extend(extended_list):
    morpho_list = list()
    for synset in extended_list:
        morpho_list += [j.synset() for j in
                        list(sum([i.derivationally_related_forms() for i in synset.lemmas()], []))]
    return morpho_list


def gloss_extend(o_sense, emb_strategy):
    """
    note: this is the main algorithm for relation exploitation,
    use different relations to retrieve bag-of-synset
    :param o_sense: the potential sense that is under expansion
    :param relation_list: all the available relations that a synset might have, except 'verb_group'
    :return: extended_list_gloss: the bag-of-synset
    """
    extended_list, combine_list = list(), [wn.synset(o_sense)]
    if 'relations' in emb_strategy:
        relation_list = ['hyponyms', 'part_holonyms', 'part_meronyms', 'member_holonyms', 'antonyms',
                     'member_meronyms', 'entailments', 'attributes', 'similar_tos', 'causes', 'pertainyms',
                     'substance_holonyms', 'substance_meronyms', 'usage_domains', 'also_sees']
        extended_list += morpho_extend([wn.synset(o_sense)])
    else:
        relation_list = ['hyponyms']

    # expand the original sense with nearby senses using all relations but hypernyms
    for index, relation in enumerate(relation_list):
        combine_list += get_related([o_sense], relation)

    # expand the original sense with in-depth hypernyms (only one branch)
    for synset in [wn.synset(o_sense)]:
        extended_list += synset.hypernyms()

    extended_list += combine_list

    return extended_list


def vector_merge(synset, key_list, emb_vecs, emb_strategy):
    new_dict = dict()
    extend_synset = gloss_extend(synset, emb_strategy)
    for key in key_list:
        sense_vec = np.array(emb_vecs[key])
        for exp_synset in extend_synset:
            distance = wn.synset(synset).shortest_path_distance(exp_synset)
            distance = distance if distance else 5
            for lemma_exp in exp_synset.lemmas():
                sense_vec += 1 / (1 + distance) * np.array(emb_vecs[lemma_exp.key()])
        sense_vec = sense_vec / np.linalg.norm(sense_vec)
        new_dict[key] = sense_vec.tolist()
    return new_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nearest Neighbors WSD Evaluation.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-emb_strategy', type=str, default='aug_gloss+r_asy+examples',
                        choices=['relations_aug_gloss+examples'],
                        help='different components to learn the basic sense embeddings', required=False)
    args = parser.parse_args()
    norm = True
    emb_strategy = 'relations_aug_gloss+examples'

    v2_path = './data/vectors/aug_gloss+examples.txt'
    logging.info('Loading %s ...' % v2_path)
    txt2_vecs = {}

    for label, vec_str in tqdm(pickle.load(open(v2_path, 'rb')).items()):
        vec = [float(v) for v in vec_str[0]]
        if norm:
            vec = normalize(vec)
        txt2_vecs[label] = vec

    synset_dict = dict()
    logging.info('Combining Vectors ...')
    for synset in wn.all_synsets():
        key_list = [i.key() for i in synset.lemmas()]
        synset_dict[synset.name()] = key_list

    vector_all = dict()
    print('synset_length: %d' % len(synset_dict))

    for synset, key_list in tqdm(list(synset_dict.items())):
        vector_all.update(vector_merge(synset, key_list, txt2_vecs, emb_strategy))

    print('key_length: %d' % len(vector_all))

    pickle.dump(vector_all, open('data/vectors/emb_wn', 'wb'), -1)