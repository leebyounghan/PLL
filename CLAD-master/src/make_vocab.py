from data_util.news20 import news20_dataset
from tokenizers import BertWordPieceTokenizer
import pdb

import os
cls = {
    "comp" : [0,1,2,3,4],
    "rec" : [5,6,7,8],
    "sci" : [9,10,11,12],
    "misc" : [13],
    "pol" : [14,15,16],
    "rel" : [17,18,19]
}

def get_data(df, name):
    return df[df['label'].isin(cls[name])]

data_train, data_test = news20_dataset()
for name in ["comp", "rec","sci","misc","pol","rel"]:
    print(f"make vocab for {name}".format(name))
    dataset = get_data(data_train, name)
    with open(f'{name}_corpus.txt'.format(name), 'w') as f:
        
        f.writelines(dataset['sentence'].tolist())
   
    tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False,
    lowercase=True)
    a = name + '_corpus.txt'
    tokenizer.train(
        [a],
        vocab_size=10000,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=1000,
        wordpieces_prefix="##")
    
    tokenizer.save_model( os.getcwd(), name)
