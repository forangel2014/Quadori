import copy
from pickletools import string4
from ngram import NGram
#import Levenshtein

def post_process_template(tB):
    if tB.endswith('.') == False:
        tB += '.'
    return tB
    # return tB.split('.')[0] + '.'


def construct_template(words, templateA, if_then=False):
    if len(words) == 2:
        # template = ['{} <mask> {}.'.format(words[0], words[1])]
        templates = [
            # '{} is <mask> {}.'.format(words[0], words[1]), 
            '{} <mask> {}.'.format(words[0], words[1]),
        ]
    elif len(words) == 1:
        templates = [
            # '{} is <mask>.'.format(words[0]),
            '{} <mask>.'.format(words[0])]

    elif len(words) == 0:
        templates = []

    if if_then:
        for word in words:
            index = templateA.index('<mask>')
            templateA = templateA[:index] + word + templateA[index + len('<mask>'):]
        templates = ['If ' + templateA + ' then ' + template for template in templates]

    return templates


def filter_words(words_prob):
    word_count = {}
    token1_count = {}
    word2_count = {}
    ret = []
    for words, prob, *_ in words_prob:
        filter_this = False

        # filter repetitive token
        token_count = {}
        for word in words:
            for token in word.split(' '):
                if token in token_count:
                    filter_this = True
                token_count[token] = 1
        if filter_this:
            prob *= 0.5

        # filter repetitive words
        if len(words) == 2 and words[0] == words[1]:
            continue

        # filter repetitive first token
        token1 = words[0].split(' ')[0]
        if token1 not in token1_count:
            token1_count[token1] = 1
        else:
            token1_count[token1] += 1
            prob /= token1_count[token1]

        for word in words:
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1
            prob /= word_count[word]

        if len(words) == 2:
            if words[1] not in word2_count:
                word2_count[words[1]] = 0
            word2_count[words[1]] += 1
            prob /= word2_count[words[1]]

        ret.append([words, prob])
    return sorted(ret, key=lambda x: x[1], reverse=True)


import math
from copy import deepcopy


def convert_for_print(arr):
    ret = deepcopy(arr)
    for i in range(len(ret)):
        ret[i][1] = round(ret[i][1], 7)
        if len(ret[i]) == 3:
            for j in range(len(ret[i][2])):
                ret[i][2][j] = round(ret[i][2][j], 7)
    return ret


def formalize_tA(tA):
    tA = tA.strip()
    if tA.endswith('.'):
        tA = tA[:-1].strip() + '.'
    else:
        tA += '.'
    tA = tA.replace(' ,', ',')
    tA = tA.replace(" '", "'")
    return tA


ngram_n = 3


def extract_similar_words(txt, words):
    max_word_length = 0
    for word in words:
        if len(word) > max_word_length:
            max_word_length = len(word)

    txt_ngrams = []
    for i in range(len(txt)):
        for j in range(i + ngram_n, min(len(txt), i + max_word_length + 5)):
            txt_ngrams.append(txt[i:j].lower())
    n = NGram(txt_ngrams, key=lambda x: x.lower(), N=ngram_n)
    ret = []
    for word in words:
        matched_word = n.find(word.lower(), 0.5)
        if matched_word is None:
            return None
        ret.append(matched_word)
    return ret


def extract_words(txt, words):
    for word in words:
        if word not in txt:
            return None
    return [word.lower() for word in words]

def jaccard(spans, text):
    ls1 = copy.deepcopy(spans)
    ls2 = copy.deepcopy(text)
    inter = 0
    union = 0
    for token in ls2:
        union += 1
        if token in ls1:
            ls1.remove(token)
            inter += 1 
    union += len(ls1)
    return inter/union if union > 0 else 0
    #return len(set(ls1).intersection(set(ls2)))/len(set(ls1).union(set(ls2)))

def clean(string):
    string = string[:-1].strip() if string.endswith('.') else string
    string = string.replace('\"', '\" ').replace(',', ' , ').replace('."', ' ."').replace('\'s', ' \'s ').replace('(', ' ( ').replace(')', ' ) ').replace('  ', ' ')
    string = string.strip()
    return string

def align(tA, text):
    
    tA = clean(tA)
    text = clean(text)

    idxs = []
    spans = tA.split('<mask>')
    text_tokens = text.split(' ')
    n = len(text_tokens)
    last_idx = 0
    for span in spans:
        if span == '':
            idxs.append([])
            continue
        span = span.strip()
        max_dis = 0
        span_tokens = span.split(' ')
        for i in range(last_idx, n):
            for j in range(i, n+1):
                dis = jaccard(span_tokens, text_tokens[i:j])
                if dis > max_dis:
                    max_dis = dis
                    idx = [i, j]
        idxs.append(idx)
        last_idx = idx[1]
    
    assert len(idxs) == 3
    idx11 = idxs[0][1] if len(idxs[0]) > 0 else 0
    idx12 = idxs[1][0]
    idx21 = idxs[1][1]
    idx22 = idxs[2][0] if len(idxs[2]) > 0 and idxs[2][0] > idx21 else n+1
    word1 = ' '.join(text_tokens[idx11:idx12])
    word2 = ' '.join(text_tokens[idx21:idx22])

    print(tA)
    print(text)
    print([word1, word2])

    return [word1, word2]