import tf_glove
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np

def read_corpus(config):
    rootpath = config['corpus']['rootpath']
    files_name = [f for f in listdir(rootpath) if isfile(join(rootpath, f))]
    corpus = []
    for fname in files_name:
        with open(join(rootpath,fname),'rb') as f:
            data = pickle.load(f)
            corpus.append(data)
    corpus = np.concatenate(corpus,axis=0)
    return corpus

def train(corpus,config):
    # corpus = [["this", "is", "a", "comment", "."], ["this", "is", "another", "comment", "."]]

    model = tf_glove.GloVeModel(embedding_size=config['model']['emb_size'],context_size=config['model']['n_gram'])
    model.fit_to_corpus(corpus)
    model.train(num_epochs=config['train']['num_epochs'])
    print(model.embedding_for("this"))

def main():
    config = {'corpus':{'rootpath':'/datastore/liu121/sentidata2/data/meituan_jieba'},
              'model':{'emb_size':'',
                       'n_gram':3},
              'train':{'num_epochs':100}
              }
    corpus = read_corpus(config)
    print(corpus.shape)
    print(corpus[0])

if __name__ == "__main__":
    main()