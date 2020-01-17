import spacy
import pickle
import re
from collections import defaultdict
from nltk.corpus import wordnet as wn
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')
for pos in ['n', 'v', 'a', 'r']:
    pos2tag = {'a': 'ADJ, AUX', 'v': 'VERB', 'r': 'ADV, ADP', 'n': 'NOUN, PROPN,'}
    vv = {i: j for i, j in pickle.load(open('./sentence_dict_%s_all' % pos, 'rb')).items()}
    sum([len(j) for j in vv.values()])/len(vv)
    valid_dict = defaultdict(list)

    # filter those sentences where the query's POS is not identical to the synset's POS
    for i, j in tqdm(vv.items()):
        for k in j:
            sentence = [sent for sent in k[1].split(',') if k[0] in sent][0]
            po_senses = set([i.name().split('.')[0] for i in wn.synsets(k[0]) if i])
            if pos in po_senses and len(po_senses) == 1:
                valid_dict[i].append((k[0], sentence, k[2]))
                continue
            tagging = nlp(sentence.lower())
            pos_list = [a.pos_ for a in tagging]
            word_list = [a.text for a in tagging]
            try:
                target = [i for i in word_list if k[0].split()[0] in i][0]

                if pos_list[word_list.index(target)] in pos2tag[pos]:
                    valid_dict[i].append((k[0], sentence, k[2]))
            except:
                valid_dict[i].append((k[0], sentence, k[2]))

    sum([len(j) for j in valid_dict.values()])/len(valid_dict)

    # filter those sentences that occur in more than one sentence set of competing synsets
    valid_dict_new = defaultdict(list)
    for i, j in valid_dict.items():
        pool = []
        if len(wn.synsets(i.split('.')[0])) > 1:
            for synset in wn.synsets(i.split('.')[0]):
                if synset.name() in valid_dict and synset.name() != i:
                    pool.extend([a[1] for a in valid_dict[synset.name()]])
        for sent in j:
            if sent[1] not in pool:
                valid_dict_new[i].append(sent)
            else:
                print(sent[1])

    sum([len(j) for j in valid_dict_new.values()])/len(valid_dict_new)

    # extrat sub-sentences and also filter those words that appear before the query
    valid_dict_reduce = defaultdict(list)
    for i, j in valid_dict_new.items():
        for sent in j:
            # we filter those sentences from the website those name contains '911'
            if '911' not in sent[2]:
                slice = re.sub('[^-\'0-9a-zA-Z]', ' ', sent[1])
                if len(slice.split()) >= 2:
                    valid_dict_reduce[i].append((sent[0], slice))

    sum([len(j) for j in valid_dict_reduce.values()])/len(valid_dict_reduce)
    valid_dict = {i: list(set([' '.join(re.sub('[^-\'0-9a-zA-Z]', ' ', k[1]).split()) for k in j])) for i, j in
                  valid_dict_reduce.items()}
    pickle.dump(valid_dict, open('./sentence_dict_%s' % pos, 'wb'), -1)
