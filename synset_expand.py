from nltk.corpus import wordnet as wn
import logging
import numpy as np
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm

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

    gloss_list, sense_inventory_final = list(), list()
    for sense in sense_inventory:
        lemma_names = [i.name().lower() for i in sense.lemmas()]
        if word.lower() in lemma_names:
            name = sense.name()
            gloss_list.append(name)
    return gloss_list


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
    relation_list = ['hyponyms', 'part_holonyms', 'part_meronyms', 'member_holonyms', 'antonyms',
                     'member_meronyms', 'entailments', 'attributes', 'similar_tos', 'causes', 'pertainyms',
                     'substance_holonyms', 'substance_meronyms', 'usage_domains', 'also_sees']
    extended_list, combine_list = list(), [wn.synset(o_sense)]

    # expand the original sense with nearby senses using all relations but hypernyms
    for index, relation in enumerate(relation_list):
        combine_list += get_related([o_sense], relation)

    # expand the original sense with in-depth hypernyms (only one branch)
    if 'r_asy' in emb_strategy:
        for synset in [wn.synset(o_sense)]:
            extended_list += [i[0] for i in synset._iter_hypernym_lists()]
    else:
        for synset in [wn.synset(o_sense)]:
            extended_list += synset.hypernyms()

    extended_list += morpho_extend([wn.synset(o_sense)])

    extended_list += combine_list

    return extended_list


def vector_expand(lemma, pos, sense_pool, emb_vecs, emb_strategy):
    new_dict = dict()
    type2pos = {1: 'n', 2: 'v', 3: 'as', 4: 'r', 5: 'as'}
    pos = type2pos[pos]
    all_name_list = retrieve_sense(lemma, pos)
    gloss_dict, synset_counter = dict(), Counter()
    for index, sense_gloss in enumerate(all_name_list):
        gloss_dict[sense_gloss] = gloss_extend(sense_gloss, emb_strategy)
    for sense_index, (sense, _) in enumerate(gloss_dict.items()):
        cur_sense_key = [i.key() for i in wn.synset(sense).lemmas() if i.name().lower() == lemma
                         and i.key() in sense_pool][0]
        sense_vec = np.array(emb_vecs[cur_sense_key])
        for exp_synset in gloss_dict[sense]:
            distance = wn.synset(sense).shortest_path_distance(exp_synset)
            distance = distance if distance else 5
            for lemma_exp in exp_synset.lemmas():
                sense_vec += 1 / (1 + distance) * np.array(emb_vecs[lemma_exp.key()])
        new_dict[cur_sense_key] = np.round(sense_vec, 6)
        sense_pool.remove(cur_sense_key)
    if sense_pool:
        # print('%s-%s-%s' % (lemma, pos, ' '.join(sense_pool)))
        for i in sense_pool:
            new_dict[i] = np.round(emb_vecs[i], 6)
    return new_dict


def expand(emb_strategy):
    norm = True
    v2_path = './data/vectors/emb_glosses_%s.txt' % emb_strategy.replace('+r_asy', '').replace('+r_sy', '').replace(
                                                                                                        '+lmms', '')
    logging.info('Loading %s ...' % v2_path)
    txt2_vecs = {}
    try:
        import pickle
        for label, vec_str in tqdm(pickle.load(open(v2_path, 'rb')).items()):
            vec = [float(v) for v in vec_str[0]]
            if norm:
                vec = normalize(vec)
            txt2_vecs[label] = vec
    except:
        with open(v2_path) as txt2_f:
            for line in tqdm(txt2_f.readlines()):
                info = line.split()
                label, vec_str = info[0], info[1:]
                vec = [float(v) for v in vec_str]
                if norm:
                    vec = normalize(vec)
                txt2_vecs[label] = vec

    sense_dict = dict()
    logging.info('Combining Vectors ...')
    for sense_key in tqdm(txt2_vecs.keys()):
        sense_pos = (sense_key.split('%')[0], sense_key.split('%')[1][0].replace('5', '3'))
        if sense_pos not in sense_dict:
            sense_dict[sense_pos] = [sense_key]
        else:
            sense_dict[sense_pos].append(sense_key)

    logging.info('Separating Senses ...')
    sense_dict_mul, sense_dict_sing = dict(), dict()
    for i, j in tqdm(sense_dict.items()):
        if len(j) > 1:
            sense_dict_mul[i] = j
        elif len(j) == 1:
            sense_dict_sing[i] = j

    print('number of cpmpeting synset groups: %d' % len(sense_dict_mul))
    vector_all = dict()
    logging.info('Main algorithm: Multi-sense vector expansion ...')
    for index, (sense_pos, key_list) in enumerate(tqdm(sense_dict_mul.items())):
        lemma, pos = sense_pos[0], int(sense_pos[1])
        vector_all.update(vector_expand(lemma, pos, key_list, txt2_vecs, emb_strategy))

    logging.info('Processing NON-Expanded Vectors ...')
    for i, j in tqdm(sense_dict_sing.items()):
        vector_all.update({j[0]: np.round(txt2_vecs[j[0]], 6)})

    return vector_all
