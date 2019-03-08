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
    # TODO: convert long word to #OTHER#
    for fname in files_name:
        data = pd.read_pickle(join(rootpath,fname))[:,1]
        for review in data:
            for sentence in review:
                sentence = sentence.split(' ')
                for i in range(len(sentence)):
                    word = sentence[i]
                    if len(list(word)) > config['corpus']['max_word_len']:
                        sentence[i] = config['corpus']['OTHER']
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

def write_embedding(config,dic):
    with open(config['output']['word_embeddings_path'],'wb') as f:
        pickle.dump({'word_to_id':dic['word_to_id'],'word_embeddings':dic['word_embeddings']},f,protocol=4)

    with open(config['output']['char_embeddings_path'],'wb') as f:
        pickle.dump({'char_to_id':dic['char_to_id'],'char_embeddings':dic['char_embeddings']},f,protocol=4)


def train(corpus,config):
    # corpus = [["this", "is", "a", "comment", "."], ["this", "is", "another", "comment", "."]]

    model = tf_glove.GloVeModel(embedding_size=config['model']['word_dim'],
                                char_embedding_size=config['model']['char_dim'],
                                context_size=config['model']['n_gram'],
                                max_word_len = config['corpus']['max_word_len'])
    model.fit_to_corpus(corpus)
    model.train(num_epochs=config['train']['num_epochs'])
    word_embeddings = model.embeddings
    padding_word_vec = np.zeros(shape=(1,config['model']['word_dim']))
    word_embeddings = np.concatenate([padding_word_vec,word_embeddings],axis=0)
    char_embeddings = model.char_embeddings
    char_to_id = model.char_to_id
    word_to_id = model.word_to_id
    write_embedding(config,{'word_to_id':word_to_id,
                            'char_to_id':char_to_id,
                            'word_embeddings':word_embeddings,
                            'char_embeddings':char_embeddings})
    # Done: extract word to id and word embedding, char to id and char embedding
    # Done: prepare character vocabulary.
    # Done: add pad to word embedding
    # fixed: max vocab size, should be all words??? max vocab size = 8 millions



def main():
    # obedient: Other must be #OTHER#
    config = {'corpus':{'corpus_path':'/datastore/liu121/sentidata2/data/meituan_jieba',
                        'corpus_name':'corpus.pkl',
                        'max_word_len':11,
                        'OTHER':'#OTHER#'},
              'model':{'word_dim':300,
                       'char_dim':200,
                       'n_gram':3},
              'train':{'num_epochs':100},
              'output':{'word_embeddings_path':'/datastore/liu121/wordEmb/aic2018cwe_wordEmb.pkl',
                        'char_embeddings_path':'/datastore/liu121/charEmb/aic2018cwe_charEmb.pkl'}
              }
    # if you decide to run prepare corpus, need to delete corpus.pkl.
    print('Prepare corpus...')
    prepare_corpus(config)
    print('Done!')
    exit()
    print('Read corpus...')
    corpus = read_corpus(config)
    print('Done!')

    # print('Analysis ...')
    # analysis(corpus)
    # print('Done!')
    print('Start training')
    train(corpus,config)
    print('Done!')

if __name__ == "__main__":
    main()