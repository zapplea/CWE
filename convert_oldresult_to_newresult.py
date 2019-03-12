import argparse

def load_data():


def convert():
    pass

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
