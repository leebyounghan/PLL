#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from torchnlp.datasets.dataset import Dataset
from sklearn.datasets import fetch_20newsgroups
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk import word_tokenize
from .misc import clean_text


import config
from .embeddings import preprocess_with_avg_bert
from .embeddings import preprocess_with_s_bert
from .embeddings import preprocess_with_avg_Glove
from .embeddings import preprocess_with_avg_fasttext
import pdb

# comp 0 1 2 3 4
# rec 5 6 7 8
# sci 9 10 11 12
#misc 13
#talk 14 15 16 
#pol 17 18 19
classes = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware','comp.windows.x',
           'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
            'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
            'misc.forsale',
            'talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast',
            'talk.religion.misc', 'alt.atheism', 'soc.religion.christian'
        ]

data_set = {'comp':[0,1,2,3,4],
            'rec':[5,6,7,8],
            'sci':[9,10,11,12],
            'misc':[13],
            'talk':[14,15,16],
            'rel':[17,18,19]}

class news20_Dataset(object):
    """Docstring for  Reuters_Dataset. """


    # 'alt.atheism': 468,'comp.graphics': 568,'comp.os.ms-windows.misc': 567,
    # 'comp.sys.ibm.pc.hardware': 578, 'comp.sys.mac.hardware': 556,
    # 'comp.windows.x': 588, 'misc.forsale': 577, 'rec.autos': 562,
    # 'rec.motorcycles': 583,'rec.sport.baseball': 572,'rec.sport.hockey': 584,
    # 'sci.crypt': 582,'sci.electronics': 574,'sci.med': 578,'sci.space': 577,
    # 'soc.religion.christian': 590, 'talk.politics.guns': 532, 'talk.politics.mideast': 547,
    # 'talk.politics.misc': 452,'talk.religion.misc': 361

    
    
    def __init__(self, root_dir: str):

        self._root_dir = root_dir
        self.classes = classes
        self.train, self.test = news20_dataset(root_dir, clean_txt=True)
       
        self.train_x = None
        self.train_y = None

        self.test_in_x = None
        self.test_in_y = None
        self.test_out_x = None
        self.test_out_y = None
        self.test_in_emb = None
        self.test_out_emb = None
        self.train_text = None
        self.test_text = None
    def preprocess_for_sentiment_understanding(self):

        which_embedding = config.embedding
        assert which_embedding in config.implemented_nlp_embeddings

        print("embedding with {} embedding".format(which_embedding))

        if which_embedding == 'avg_glove':
            self.train_x, self.train_y, self.test_x, self.test_y, self.train_text, self.test_text =\
                                    preprocess_with_avg_Glove(self.train, self.test)
        elif which_embedding == 'avg_bert':
            self.train_x, self.train_y, self.test_x, self.test_y =\
                                    preprocess_with_avg_bert(self.train, self.test)
        elif which_embedding == 's_bert':
            self.train_x, self.train_y, self.test_x, self.test_y, self.train_text, self.test_text =\
                                    preprocess_with_s_bert(self.train, self.test)
        elif which_embedding == 'avg_fasttext':
            self.train_x, self.train_y, self.test_x, self.test_y, self.train_text, self.test_text =\
                                    preprocess_with_avg_fasttext(self.train, self.test)
        else:
            pass #no emb            
        return self.train_x, self.train_y, self.test_x, self.test_y, self.train_text, self.test_text


    # get label 0 for normal class and 1 for the rest
    def get_binary_labeled_data(self):
        #TODO: change the normal class index with configuration

        
        self.train_x, _, _, _, label,self.train_y = divide_data_label(self.train_x,
                                                             self.train_y,
                                                             train=True)

        if config.classifier_type in ['bert','DEC_bert',"bert_scr"]:
            print("we in bert")
            self.train_text, _, _, _, _,c_in_label = divide_data_label(self.train_text,label , train = True)
            self.test_in_x, self.test_in_y, self.test_out_x, self.test_out_y, _,_ = divide_data_label(self.test_text,
                                                                       self.test_y, train=False)
            self.test_in_emb, _, self.test_out_emb, _, _,_ = divide_data_label(self.test_x,
                                                                                             self.test_y,
                                                                                             train=False)
            self.test_in_emb = torch.tensor(self.test_in_emb)
            self.test_out_emb = torch.tensor(self.test_out_emb)
        else:
            self.test_in_x, self.test_in_y, self.test_out_x, self.test_out_y, _,_ = divide_data_label(self.test_x,
                                                                                             self.test_y,
                                                                                             train=False)

            
            self.test_in_x = torch.tensor(self.test_in_x)
            self.test_out_x = torch.tensor(self.test_out_x)
        
            

        self.train_x = torch.tensor(self.train_x)
        self.train_y = torch.tensor(self.train_y)
        dataset = {"train_x": self.train_x, 
                   "train_y": self.train_y,
                   "test_in": self.test_in_x, 
                   "test_out": self.test_out_x,
                   "test_in_emb" : self.test_in_emb,
                   "test_out_emb" : self.test_out_emb,
                   "train_text": self.train_text,}

        return dataset


def divide_data_label(dataset, label, train=False):
    in_data = []
    out_data = []
    in_labels = []
    out_labels = []
    #c_out_labels = []
    c_in_labels = []
    
    for i, _d in enumerate(dataset):
        data_x = _d
        data_y = label[i]
        
        if (data_y in config.normal_class_index_list):
            in_data.append(data_x)
            in_labels.append(0)
            c_in_labels.append(config.normal_class_index_list.index(data_y))
        else:
            if (train): continue
            else:
                out_data.append(data_x)
                out_labels.append(1)
                #c_out_labels.append(data_y) # for data chek
     
   
    return in_data, in_labels, out_data, out_labels, label, c_in_labels 


def news20_dataset(directory='../data', clean_txt = False):

    train = True
    test = True

    if directory not in nltk.data.path:
        nltk.data.path.append(directory)

    dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

    ret = []
    splits = [
        split_set
        for (requested, split_set) in [(train, 'train'), (test, 'test')]
        if requested
    ]

    for split_set in splits:

        dataset = fetch_20newsgroups(data_home=directory, subset=split_set, remove=('headers', 'footers', 'quotes'))
        examples = []

        for id in range(len(dataset.data)):
            if clean_txt:
                text = clean_text(dataset.data[id])
            else:
                text = ' '.join(word_tokenize(dataset.data[id]))
            label = dataset.target_names[int(dataset.target[id])]

            if text:
                examples.append({
                    'text': text,
                    'label': label
                })

        ret.append(Dataset(examples))

    ret_sentences = []
    ret_labels = []

    for ret_ in ret:

        sentence = []
        label = []

        for i, label_ in enumerate(ret_['label']):

            label_string = label_
            if label_string in classes:
                label.append(classes.index(label_string))
                sentence.append(ret_['text'][i])

        ret_sentences.append(sentence)
        ret_labels.append(label)

    train = pd.DataFrame({
        'sentence': ret_sentences[0],
        'label': ret_labels[0]
    })
    #  train, val = np.split(train.sample(frac=1), [int(
    #      (0.9) * len(train))])  # split 9:1 = train:val
    test = pd.DataFrame({'sentence': ret_sentences[1], 'label': ret_labels[1]})

    return train, test
