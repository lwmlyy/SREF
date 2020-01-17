import requests
import json
import re
import urllib
import execjs
import time
from fake_useragent import UserAgent
import os
from collections import defaultdict
import pickle
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
import spacy
nlp = spacy.load('en_core_web_sm')
ua = UserAgent()


def get_code(headers, source):
    html = requests.get('https://fanyi.baidu.com', headers=headers)
    html.encoding = 'utf-8'

    matches = re.findall("window.gtk = '(.*?)';", html.text, re.S)
    for match in matches:
        gtk = match

    if gtk == "":
        print('Get gtk fail.')
        exit()

    matches = re.findall("token: '(.*?)'", html.text, re.S)
    for match in matches:
        token = match

    if token == "":
        print('Get token fail.')
        exit()

    # 计算 sign
    signCode = 'function a(r,o){for(var t=0;t<o.length-2;t+=3){var a=o.charAt(t+2);a=a>="a"?a.' \
               'charCodeAt(0)-87:Number(a),a="+"===o.charAt(t+1)?r>>>a:r<<a,r="+"===o.charAt(t)?r+a&4294967295:r^a}' \
               'return r}var C=null;var hash=function(r,_gtk){var o=r.length;o>30&&(r=""+r.substr(0,10)+' \
               'r.substr(Math.floor(o/2)-5,10)+r.substr(-10,10));var t=void 0,t=null!==C?C:(C=_gtk||"")||""' \
               ';for(var e=t.split("."),h=Number(e[0])||0,i=Number(e[1])||0,d=[],f=0,g=0;g<r.length;g++)' \
               '{var m=r.charCodeAt(g);128>m?d[f++]=m:(2048>m?d[f++]=m>>6|192:(55296===(64512&m)&&g+1<r.' \
               'length&&56320===(64512&r.charCodeAt(g+1))?(m=65536+((1023&m)<<10)+(1023&r.charCodeAt(++g)' \
               '),d[f++]=m>>18|240,d[f++]=m>>12&63|128):d[f++]=m>>12|224,d[f++]=m>>6&63|128),d[f++]=63&m|128)' \
               '}for(var S=h,u="+-a^+6",l="+-3^+b+-f",s=0;s<d.length;s++)S+=d[s],S=a(S,u);return S=a(S,l),S^' \
               '=i,0>S&&(S=(2147483647&S)+2147483648),S%=1e6,S.toString()+"."+(S^h)}'

    sign = execjs.compile(signCode).call('hash', source, gtk)
    fromLanguage = 'en'
    toLanguage = 'zh'
    v2transapi = 'https://fanyi.baidu.com/v2transapi?from=%s&to=%s&query=%s' \
                 '&transtype=translang&simple_means_flag=3&sign=%s&token=%s' % (
                     fromLanguage, toLanguage, urllib.parse.quote(source), sign, token)
    return v2transapi


def get_examples(source, record):
    headers = {
        "User-Agent": ua.random,
        "Cookie": "BAIDUID=52B3700AB6F5B4062AF02C3190F2E976:FG=1; BIDUPSID=52B3700AB6F5B4062AF02C3190F2E976; "
                  "PSTM=1544531074; MCITY=-%3A",
    }

    success = False
    examples, failed = list(), list()
    while not success:
        try:
            v2transapi = get_code(headers, source)
            translate_result = json.loads(requests.get(v2transapi, headers=headers).text)
            if translate_result["liju_result"]['double']:
                for sentences in json.loads(translate_result["liju_result"]['double']):
                    web_source = sentences[2]
                    if 'provided by jukuu' not in web_source:
                        eng_example = ' '.join([i[0] for i in sentences[0]])
                        if source not in eng_example:
                            continue
                        else:
                            # extract sub-sentence where the query locates
                            eng_example = ' '.join([i.strip() for i in eng_example.split(',') if
                                                    source in i and len(i.split()) <= 15])
                        if eng_example:
                            examples.append((source, eng_example, web_source))
                print('%s-succeeded!' % record)
                success = True
            else:
                print('%s-no content for the query!' % record)
                success = True
            time.sleep(0.5)
        except:
            print('%s-failed!' % record)
            time.sleep(0.5)
            success = True
            failed.append(source)

    return examples, failed


def verb_phrase(definition):
    # obtain unigram, bigram, trigram and others from a synset's definition
    # mainly based on the length, the location of ';', 'or' and others, and POS information
    def_words = definition.split()
    if len(def_words) <= 5:
        if 'or' in def_words:
            if len(def_words) == 3 and ';' not in definition:
                def_words.remove('or')
                return def_words
            elif len(def_words) == 4 and ';' not in definition:
                if def_words[1] == 'or':
                    return [' '.join(def_words[2:]), def_words[0] + ' ' + def_words[3]]
                else:
                    return [' '.join(def_words[:2]), def_words[0] + ' ' + def_words[3]]
            else:
                if len(definition) - len(definition.replace(',', '')) == 2:
                    def_words = definition.replace(',', '').split()
                    center = (def_words[0], 0) if def_words[-2] == 'or' else (def_words[-1], -1)
                    def_words = [i for i in def_words if i != 'or']
                    return [center[0] + ' ' + i if center[1] == 0 else i + ' ' + center[0] for i in def_words if i != center[0]]
                elif len(definition) - len(definition.replace(',', '')) == 1:
                    pos_result = pos_tag(word_tokenize(definition))
                    def_words = [i for i, j in pos_result if j[0] in 'NVJRC']
                    center = (def_words[0], 0) if def_words[-2] == 'or' else (def_words[-1], -1)
                    return [center[0] + ' ' + i if center[1] == 0 else i + ' ' + center[0] for i in def_words if
                            i != center[0]]
                elif len(definition) - len(definition.replace(',', '')) > 2:
                    def_words.remove('or')
                    return def_words
                else:
                    if ';' in definition:
                        spans = []
                        def_span = definition.split(';')
                        for span in def_span:
                            if ' or ' not in span:
                                spans.append(span)
                            else:
                                def_words = span.split()
                                if len(def_words) == 3:
                                    spans.extend(span.split(' or '))
                                else:
                                    if def_words[1] == 'or':
                                        spans.extend([' '.join(def_words[2:]), def_words[0] + ' ' + def_words[3]])
                                    else:
                                        spans.extend([' '.join(def_words[:2]), def_words[0] + ' ' + def_words[3]])
                        return spans
                    else:
                        pos_result = [token.pos_ for token in nlp(definition)]
                        if 'of or relating to' in definition:
                            return []
                        elif pos_result[1] == 'ADP':
                            return [definition.split(' or ')[1]]
                        else:
                            def_words = definition.replace(',', '').split()
                            if def_words[2] != 'or':
                                if def_words[1] == 'or':
                                    return [' '.join(def_words[2:]), def_words[0] + ' ' + ' '.join(def_words[3:])]
                                else:
                                    return [' '.join(def_words[:3]), ' '.join(def_words[:2]) + ' ' + def_words[4]]
                            else:
                                if pos_result[1] == pos_result[3]:
                                    if pos_result[2] == pos_result[4]:
                                        phrase = definition.split(' or ')
                                    else:
                                        phrase = [' '.join([def_words[0], def_words[1], def_words[-1]]),
                                                  ' '.join([def_words[0], def_words[3], def_words[-1]])]
                                else:
                                    phrase = [' '.join(def_words[:def_words.index('or')]), def_words[0] +
                                              ' ' + ' '.join(def_words[def_words.index('or')+1:])]
                                return phrase
        else:
            if len(def_words) <= 4 or ';' in definition:
                return definition.replace(',', '').split(';')
            else:
                return []
    else:
        return []


# the whole process might take a long time for all POS, better deploy the process of each POS in separate machines
# for noun synsets, we only deal with those that does not have example sentences
for pos in ['n', 'v', 'a', 'r']:
    non_retreive = list()
    type2pos = {1: 'n', 2: 'v', 3: 'a', 4: 'r', 5: 'a'}
    all_synsets = [i.name() for i in wn.all_synsets(pos) if
                   len(wn.synsets(i.name().split('.')[0], type2pos[int(i.lemmas()[0].key().split('%')[1][0])])) > 0]
    if os.path.exists('./sentence_dict_%s_new' % pos):
        sentence_dict = {i: j for i, j in pickle.load(open('./sentence_dict_%s_new' % pos, 'rb')).items()}
        non_retreive = [i for i in sentence_dict.keys()]
    else:
        sentence_dict = defaultdict(list)
        non_retreive = all_synsets

    loop_bool = True
    while loop_bool:
        failed_list = list()
        for index, synset in enumerate(tqdm(non_retreive)):
            synset = wn.synset(synset)
            if pos == 'n' and synset.examples():
                valid_span = []
            else:
                definition = synset.definition()
                valid_span = verb_phrase(definition)

            succeed_list = list()
            if len(wn.synsets(synset.name().split('.')[0], pos)) == 1:
                valid_span.append(synset.name().split('.')[0])
            if valid_span:
                failed_list.append(synset)
                valid_span = [i.strip() for i in valid_span]
                for span in valid_span:
                    span = ' '.join(re.sub('[^-\'0-9a-zA-Z]', ' ', span).split())
                    retrieved, failed = get_examples(span, '%s-%s-%s' % (str(index), synset.name(), span), pos)
                    succeed_list.append(False if failed else True)
                    if retrieved:
                        sentence_dict[synset.name()].extend(retrieved)
                if any(succeed_list):
                    failed_list.remove(synset)
            else:
                pass
            pickle.dump(sentence_dict, open('./sentence_dict_%s_all' % pos, 'wb'), -1)
        if not failed_list:
            loop_bool = False
        else:
            non_retreive = [i.name() for i in failed_list]
