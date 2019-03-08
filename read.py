import pickle

with open('/datastore/liu121/charEmb/aic2018cwep_charEmb.pkl','rb') as f:
    dic = pickle.load(f)
    print(dic.keys())
    print(dic['char_to_id'])
    print(dic['char_embeddings'])

with open('/datastore/liu121/wordEmb/aic2018cwep_wordEmb.pkl','rb') as f:
    dic = pickle.load(f)
    print(dic.keys())
    print(dic['word_to_id'])
    print(dic['word_embeddings'])

    #fixme: char['#PAD#'] is not [0,....]
    # fixme: word_to_id['#PAD#'] doesn't exits