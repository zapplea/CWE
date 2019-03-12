import pickle

with open('/datastore/liu121/charEmb/aic2018cwep_charEmb.pkl','rb') as f:
    dic = pickle.load(f)
    print(dic.keys())
    print(dic['char_to_id'])
    print(dic['char_embeddings'])
    print('char dic len: ',len(dic['char_to_id']))
    print('char embedding len: ',len(dic['char_embeddings']))
    exit()
with open('/datastore/liu121/wordEmb/aic2018cwep_wordEmb.pkl','rb') as f:
    dic = pickle.load(f)
    print(dic.keys())
    print(dic['word_to_id'])
    print(dic['word_embeddings'])
    print('word dic len: ',len(dic['word_to_id']))
    print('word embedding len',len(dic['word_embeddings']))

    # fixed: char['#PAD#'] is not [0,....]
    # fixme: word_to_id['#PAD#'] doesn't exits