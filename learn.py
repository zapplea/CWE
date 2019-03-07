import tf_glove
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
import pandas as pd
import re
import getpass
username = getpass.getuser()

def resave():
    rootpath = '/home/yibing/Documents/data/nlp/meituan_jieba'
    files_name = [f for f in listdir(rootpath) if isfile(join(rootpath, f))]
    for fname in files_name:
        data = pd.read_pickle(join(rootpath, fname)).values
        new_rootpath = '/home/yibing/Documents/data/nlp/new_meituan_jieba'
        with open(join(new_rootpath,fname),'wb') as f:
            pickle.dump(data,f,protocol=4)

def prepare_corpus(config):
    rootpath = config['corpus']['corpus_path']
    files_name = [f for f in listdir(rootpath) if isfile(join(rootpath, f))]
    corpus = []
    # TODO: eliminate long word
    for fname in files_name:
        data = pd.read_pickle(join(rootpath,fname))[:,1]
        for review in data:
            for sentence in review:
                is_exceedmax = False
                sentence = sentence.split(' ')
                for word in sentence:
                    if len(list(word)) > config['corpus']['max_word_len']:
                        is_exceedmax = True
                        break
                if is_exceedmax:
                    continue
                corpus.append(sentence)
    print('corpus length: ',len(corpus))
    print('sample:\n',corpus[77])
    with open(join(rootpath,config['corpus']['corpus_name']),'wb') as f:
        pickle.dump(corpus,f)

def analysis(corpus):
    max_len = 11
    extra_word_dic = {}
    extra_count = 0
    for sentence in corpus:
        for word in sentence:
            if len(list(word))>max_len:
                extra_count+=1
                if word not in extra_word_dic:
                    extra_word_dic[word]=[sentence]
                else:
                    extra_word_dic[word].append(sentence)
                break
    lang_count = 0
    for sentence in corpus:
        for word in sentence:
            if re.search(u'[a-zA-Z0-9]*',word):
                lang_count+=1
                break
    for key in extra_word_dic:
        print(extra_word_dic[key])
        print('=================')
    print('etrax total nums: ',extra_count)
    print('lang total nums: ',lang_count)


def read_corpus(config):
    with open(join(config['corpus']['corpus_path'],config['corpus']['corpus_name']),'rb') as f:
        corpus = pickle.load(f)
    return corpus

def train(corpus,config):
    # corpus = [["this", "is", "a", "comment", "."], ["this", "is", "another", "comment", "."]]

    model = tf_glove.GloVeModel(embedding_size=config['model']['emb_size'],context_size=config['model']['n_gram'])
    model.fit_to_corpus(corpus)
    model.train(num_epochs=config['train']['num_epochs'])
    # TODO: extract word to id and word embedding

def main():
    config = {'corpus':{'corpus_path':'/datastore/liu121/sentidata2/data/meituan_jieba',
                        'corpus_name':'corpus.pkl',
                        'max_word_len':11},
              'model':{'emb_size':'',
                       'n_gram':3},
              'train':{'num_epochs':100}
              }
    print('Prepare corpus...')
    prepare_corpus(config)
    print('Done!')
    exit()
    print('Read corpus...')
    corpus = read_corpus(config)
    print('Done!')

    print('Analysis ...')
    analysis(corpus)
    print('Done!')
    # train(corpus,config)

if __name__ == "__main__":
    main()