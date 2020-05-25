# A Synset Relation-enhanced Framework with a Try-again Mechanism for Word Sense Disambiguation

This repository is the open source code for SREF, a Knowledge-Enhanced Sense Embedding method. Many modules come from [LMMS](https://github.com/danlou/lmms). We thank the authors for opening their valuable modules.

### Quick Evaluation
For a quick evaluation of our systems' (SREF<sub>kb</sub>, SREF<sub>sup</sub>) results on five WSD datasets (SE2, SE3, SE07, SE13 and SE15) and the combined dataset (ALL), run command.sh on your Linux machine or use a Git bash tool on Windows. 

## Table of Contents
- [Requirements](#Requirements)
- [Gloss_Augment](#Crawl-augmented_gloss)
- [Loading BERT](#Loading-BERT)
- [Basic Sense Embeddings](#basic-sense-embeddings)
- [Sense Embeddings Enhancement](#sense embeddings enhancement)
- [WSD Evaluation](#WSD-evaluation)


### Requirements

The whole project is built under python 3.6, with anaconda providing the basic packages. Additional packages are included in the requirements.txt, which can be installed with the following code:

```bash
pip install -r requirements.txt
```

The WordNet package for NLTK isn't installed by pip, but we can install it easily with:

```bash
$ python -c "import nltk; nltk.download('wordnet')"
```

We use WordNet-3.0 for the entire experiment. NOTE: the online version of WordNet is the latest version-3.1, which returns different results from those from WordNet-3.0 from python.


The basic sense representations are learned from BERT<sub>LARGE_CASED</sub>, so download it with the following code:

```bash
$ cd data/bert  # from repo home
$ wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip
$ unzip cased_L-24_H-1024_A-16.zip
```

### Crawl augmented_gloss
Use example_expand.py to crawl sentences from a translation website. Note that we do not use any of the translation results or other information provided by the website. After this process, run example_filter.py to filter those noisy sentences.
NOTE: this process takes a considerably lone time, especially for nouns. One can run four process (synsets for 4 POS) on different machines. Alternatively, you can download the processed files from [here](https://drive.google.com/open?id=1qvQ-y2ylD8vLqSrHLPLJkq3ugAjxVOrD) (put them in ./).
```bash
$ python example_expand.py
$ python example_filter.py
```

### Loading BERT

The project relies on [bert-as-service](https://github.com/hanxiao/bert-as-service), to retrieve BERT embeddings. This process requires a GPU devise which has at least 5GB memory (Large memory, Faster processing). You should not set -max_seq_len to a large number unless it is necessary, because it slows down the process dramatically (a lot of padding of 0s and thus unnecessary calculations).

####Parameter choice:  
- -pooling_strategy REDUCE_MEAN, for basic sense embedding learning
- -pooling_strategy NONE, for evaluation
```bash
$ bert-serving-start -pooling_strategy REDUCE_MEAN -model_dir data/bert/cased_L-24_H-1024_A-16 -pooling_layer -1 -2 -3 -4 -max_seq_len NONE -max_batch_size 32 -num_worker=1 -device_map 0 -cased_tokenization
$ bert-serving-start -pooling_strategy NONE -model_dir data/bert/cased_L-24_H-1024_A-16 -pooling_layer -1 -2 -3 -4 -max_seq_len NONE -max_batch_size 32 -num_worker=1 -device_map 0 -cased_tokenization
```

You should see the following message when the server is ready:

```bash
I:VENTILATOR:[__i:_ru:163]:all set, ready to serve request!
```

This process should be left open until we start the evaluation process. If you really want to finish the whole experiment with one session, you can use the following code:

```bash
$ nohup bert-serving-start -pooling_strategy REDUCE_MEAN -model_dir data/bert/cased_L-24_H-1024_A-16 -pooling_layer -1 -2 -3 -4 -max_seq_len NONE -max_batch_size 32 -num_worker=1 -device_map 0 -cased_tokenization > nohup.out &
```

When you start with the evaluation process, use the following code to kill the above server. NOTE: this will kill all processes that are related to 'bert-serving-start'
```bash
$ ps -ef  grep bert-serving-start  grep -v grep  awk '{print "kill -9 "$2}'  sh
```

### basic sense embeddings
When the BERT server is ready, you should run emb_glosses.py to get the basic sense embeddings from the sense gloss, augmented sentences, and example sentences (usage). For SREF<sub>kb</sub>, run the following code:
#####Parameter choice:  
- -emb_strategy aug_gloss+examples, SREF<sub>enhanced</sub>, SREF<sub>sup</sub>
```bash
$ python emb_glosses.py -emb_strategy aug_gloss+examples
```

Also, we provided the file: [aug_gloss+examples](https://drive.google.com/open?id=1Ef7--gC-jJXXjn8Dryp4umO6WnKQXvsD) so that you can implement the following codes conveniently (put them in /data/vectors).

### sense embeddings enhancement
run synset_expand.py to enhance the basic sense embeddings.
```bash
$ python synset_expand.py
```

### WSD evaluation
Before evalution, you should stop the previous bert-as-server process and starts a new one with the parameter **-pooling_strategy** set to **NONE**.  
When the basic embeddings and BERT server are ready, run eval_nn.py to evaluate our method. Note that we merge the synset expansion (synset_expand.py) algorithm in this file as a function. You should get the following results for the knowledge-based system  

#####Parameter choice:  
- -emb_strategy aug_gloss+examples, SREF<sub>kb</sub>
- -emb_strategy aug_gloss+examples+lmms, SREF<sub>sup</sub>
- -sec_wsd False to disable the second wsd/ try-again mechanism

```bash
$ python emb_glosses.py -emb_strategy aug_gloss+r_asy
```
    
For SREF<sub>sup</sub>, you need to get the [LMMS supervised sense embeddings](https://drive.google.com/open?id=13lD2t3aj-n22fvv77MWMTn67pZw196yI) by train.py. It relies on SemCor to learn sense embeddings as a starting point. By running eval_nn.py, you should get the following results.

```bash
$ python emb_glosses.py -emb_strategy aug_gloss+r_sy+examples+lmms
```

| |SE2|SE3|SE07|SE13|SE15|ALL|
|----------------|----|----|----|----|----|-----------------|
|SREFkb|72.7|71.5|61.8|76.4|79.5|73.5|
|SREFsup|78.6|76.6|72.1|78|80.5|77.8 |  
