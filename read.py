import pickle

with open('/datastore/liu121/charEmb/aic2018cwep_charEmb.pkl','rb') as f:
    dic = pickle.load(f)
    print(dic.keys())
    print(dic['char_to_id'])
    print(dic['char_embeddings'])
    print('char dic len: ',len(dic['char_to_id']))
    print('char embedding len: ',len(dic['char_embeddings']))

with open('/datastore/liu121/wordEmb/aic2018cwe_wordEmb.pkl_v2','rb') as f:
    dic = pickle.load(f)
    print(dic.keys())
    print(dic['word_to_id'])
    print(dic['word_embeddings'])
    print('word dic len: ',len(dic['word_to_id']))
    print('word embedding len',len(dic['word_embeddings']))

    # fixed: char['#PAD#'] is not [0,....]
    # fixed: word_to_id['#PAD#'] doesn't exits