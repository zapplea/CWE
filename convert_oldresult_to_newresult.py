import argparse
import pickle

def convert(fname):
    with open(fname, 'rb') as f:
        dic = pickle.load(f)
        word_dic = dic['word_to_id']
        word_embeddings = dic['word_embeddings']
        print('id of #OTHER#: ',word_dic['#OTHER#'])
        word_dic['#UNK#'] = word_dic.pop('#OTHER#')
        print('id of #UNK#: ',word_dic['#UNK#'])
        print('#OTHER# doesn\'t exit: ',str('#OTHER#' not in word_dic))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['cwe', 'cwep', 'cwel'], default='cwe')
    args = parser.parse_args()
    output_path = {'cwe': {'word_embeddings_path': '/datastore/liu121/wordEmb/aic2018cwe_wordEmb.pkl',
                           'char_embeddings_path': '/datastore/liu121/charEmb/aic2018cwe_charEmb.pkl'},
                   'cwep': {'word_embeddings_path': '/datastore/liu121/wordEmb/aic2018cwep_wordEmb.pkl',
                            'char_embeddings_path': '/datastore/liu121/charEmb/aic2018cwep_charEmb.pkl'},
                   'cwel': {'word_embeddings_path': '/datastore/liu121/wordEmb/aic2018cwel_wordEmb.pkl',
                            'char_embeddings_path': '/datastore/liu121/charEmb/aic2018cwel_charEmb.pkl'}}
    convert(output_path[args.mode]['word_embeddings_path'])