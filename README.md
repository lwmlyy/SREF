# Knowledge-Enhanced Word Sense Disambiguation: Language Model and Beyond

This repository is the open source code for KESE, a Knowledge-Enhanced Sense Embedding method. Many modules come from [LMMS](https://github.com/danlou/lmms)

### Quick Evaluation
For a quick evaluation of our systems' (KESE<sub>base</sub>, KESE<sub>enhanced</sub>, KESE<sub>sup</sub>) results on five WSD datasets (SE2, SE3, SE07, SE13 and SE15) and the combined dataset (ALL), run command.sh on your Linux machine or use a Git bash tool on Windows. 

## Table of Contents
- [Requirements](#Requirements)
- [Augmented_gloss](#Crawl-augmented_gloss)
- [Loading BERT](#Loading-BERT)
- [Basic sense embeddings](#basic-sense-embeddings)
- [WSD evaluation](#WSD-evaluation)


### Requirements

The whole project is built under python 3.6, with anaconda providing the basic packages. Additional packages are included in the requirements.txt, which can be installed with the following code:

```bash
pip install -r requirements.txt
```

The WordNet package for NLTK isn't installed by pip, but we can install it easily with:

```bash
$ python -c "import nltk; nltk.download('wordnet')"
```

We use WordNet-3.0 for the entire experiments. NOTE: the online version of WordNet is the latest version-3.1, which returns different results from those from WordNet-3.0 from python.


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
#####Parameter choice:  
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
$ ps -ef | grep bert-serving-start | grep -v grep | awk '{print "kill -9 "$2}' | sh
```

### basic sense embeddings
When the BERT server is ready, you should run emb_glosses.py to get the basic sense embeddings from the sense gloss, augmented sentences, and example sentences (usage). For KESE<sub>base</sub>, run the following code:
#####Parameter choice:  
- -emb_strategy aug_gloss, KESE<sub>base</sub>
- -emb_strategy aug_gloss+examples, KESE<sub>enhanced</sub>, KESE<sub>sup</sub>
```bash
$ python emb_glosses.py -emb_strategy aug_gloss
```

You should get a file (in python pickle format, faster to save and load) for processes with each parameter. Also, we provided these files: [aug_gloss](https://drive.google.com/open?id=1h8mpFLCfe095URAFPxVuciAO6nf3y-Mq), [aug_gloss+examples](https://drive.google.com/open?id=1E28mw0T-5vI4FpzyJ9Eb9aEawUYWTFao) so that you can implement the following codes conveniently (put them in /data/vectors).


### WSD evaluation
Before evalution, you should stop the previous bert-as-server process and starts a new one with the parameter **-pooling_strategy** set to **NONE**.  
When the basic embeddings and BERT server are ready, run eval_nn.py to evaluate our method. Note that we merge the synset expansion (synset_expand.py) algorithm in this file as a function. You should get the following results for two knowledge-based systems  

#####Parameter choice:  
- -emb_strategy aug_gloss+r_asy, KESE<sub>base</sub>
- -emb_strategy aug_gloss+r_sy+examples, KESE<sub>enhanced</sub>
- -emb_strategy aug_gloss+r_sy+examples+lmms, KESE<sub>sup</sub>

```bash
$ python emb_glosses.py -emb_strategy aug_gloss+r_asy
```

KESE<sub>base</sub>
>>  semeval2007 P= 58% R= 58% F1= 58%  
    semeval2013 P= 72.5% R= 72.5% F1= 72.5%  
    semeval2015 P= 73.5% R= 73.5% F1= 73.5%  
    senseval2 P= 68.9% R= 68.9% F1= 68.9%  
    senseval3 P= 66.2% R= 66.2% F1= 66.2%  
    ALL P= 69% R= 69% F1= 69%  
    
KESE<sub>enhanced</sub>
>>  semeval2007 P= 63.5% R= 63.5% F1= 63.5%  
    semeval2013 P= 74.3% R= 74.3% F1= 74.3%  
    semeval2015 P= 76.3% R= 76.3% F1= 76.3%  
    senseval2 P= 71.6% R= 71.6% F1= 71.6%  
    senseval3 P= 69.4% R= 69.4% F1= 69.4%  
    ALL P= 71.8% R= 71.8% F1= 71.8%  
    
For KESE<sub>sup</sub>, you need to get the [LMMS supervised sense embeddings](https://drive.google.com/open?id=1JwkqCRfPSODk5ePwcuFsTK_bxvd9YI9c) by train.py and extend.py. It relies on SemCor to learn sense embeddings as a starting point. By running eval_nn.py, you should get the following results.

```bash
$ python emb_glosses.py -emb_strategy aug_gloss+r_sy+examples+lmms
```

KESE<sub>enhanced</sub>
>>  semeval2007 P= 72.5% R= 72.5% F1= 72.5%  
    semeval2013 P= 78.1% R= 78.1% F1= 78.1%  
    semeval2015 P= 79% R= 79% F1= 79%  
    senseval2 P= 78.6% R= 78.6% F1= 78.6%  
    senseval3 P= 76.4% R= 76.4% F1= 76.4%  
    ALL P= 77.6% R= 77.6% F1= 77.6%  
