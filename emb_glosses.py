import time
import argparse
import logging
from collections import defaultdict

import numpy as np
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from tqdm import tqdm

from bert_as_service import bert_embed_sents
# from bert_as_service import bert_embed


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def wn_synset2keys(synset):
    if isinstance(synset, str):
        synset = wn.synset(synset)
    return list(set([lemma.key() for lemma in synset.lemmas()]))


def fix_lemma(lemma):
    return lemma.replace('_', ' ')


def get_sense_data(emb_strategy):
    data = []
    import pickle
    name = locals()
    for pos in ['n', 'r', 'v', 'a']:
        try:
            name['%s_example' % pos] = pickle.load(open('./sentence_dict_%s' % pos, 'rb'))
            name['%s_example' % pos] = {i: [k for k in j] for i, j in name['%s_example' % pos].items() if j}
            print('%s sentences loaded!' % pos)
        except:
            name['%s_example' % pos] = {}
    type2pos = {1: 'n', 2: 'v', 3: 'a', 4: 'r', 5: 'a'}
    for index, synset in enumerate(wn.all_synsets()):
        all_lemmas = [fix_lemma(lemma.name()) for lemma in synset.lemmas()]
        gloss = ' '.join(word_tokenize(synset.definition()))
        ty = int([i.key() for i in synset.lemmas()][0].split('%')[1][0])
        if synset.name() in name['%s_example' % type2pos[ty]]:
            examples = ' '.join(word_tokenize(' '.join(name['%s_example' % type2pos[ty]][synset.name()])))
        else:
            examples = ''
        if 'examples' in emb_strategy:
            examples += ' '.join(word_tokenize(' '.join(synset.examples())))
        for lemma in synset.lemmas():
            lemma_name = fix_lemma(lemma.name())
            d_str = lemma_name + ' - ' + ' , '.join(all_lemmas) + ' - ' + gloss + examples
            data.append((synset, lemma.key(), d_str))

    data = sorted(data, key=lambda x: x[0])
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates sense embeddings based on glosses and lemmas.')
    parser.add_argument('-batch_size', type=int, default=64, help='Batch size (BERT)', required=False)
    parser.add_argument('-emb_strategy', type=str, default='aug_gloss',
                        choices=['aug_gloss', 'aug_gloss+examples'],
                        help='different components to learn the basic sense embeddings', required=False)
    parser.add_argument('-out_path', help='Path to resulting vector set', required=False,
                        default='data/vectors/emb_glosses_%s.txt')
    args = parser.parse_args()
    pooling_strategy = 'REDUCE_MEAN' # important parameter to replicate results using bert-as-service

    logging.info('Preparing Gloss Data ...')
    glosses = get_sense_data(args.emb_strategy)
    glosses_vecs = defaultdict(list)

    logging.info('Embedding Senses ...')
    t0 = time.time()
    for batch_idx, glosses_batch in enumerate(tqdm(chunks(glosses, args.batch_size))):
        dfns = [e[-1] for e in glosses_batch]

        dfns_bert = bert_embed_sents(dfns, strategy=pooling_strategy)

        for (synset, sensekey, dfn), dfn_bert in zip(glosses_batch, dfns_bert):
            dfn_vec = dfn_bert[1]
            glosses_vecs[sensekey].append(dfn_vec)

        t_span = time.time() - t0
        n = (batch_idx + 1) * args.batch_size
        logging.info('%d/%d at %.3f per sec' % (n, len(glosses), n/t_span))

    logging.info('Writing Vectors %s ...' % args.out_path)
    import pickle
    pickle.dump(glosses_vecs, open(args.out_path % str(args.emb_strategy), 'wb'), -1)
