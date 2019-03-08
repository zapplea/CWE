import pickle

with open('/datastore/liu121/charEmb/aic2018cwe_charEmb.pkl','rb') as f:
    dic = pickle.load(f)
    print(dic.keys())

with open('/datastore/liu121/wordEmb/aic2018cwe_wordEmb.pkl','rb') as f:
    dic = pickle.load(f)
    print(dic.keys())
    print(dic['char_to_id'])
    print(dic['char_embeddings'])