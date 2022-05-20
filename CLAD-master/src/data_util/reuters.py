#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from torchnlp.datasets.dataset import Dataset
from nltk.corpus import reuters


from nltk import word_tokenize
from .misc import clean_text

import nltk

import config
from .embeddings import preprocess_with_avg_bert
from .embeddings import preprocess_with_s_bert
from .embeddings import preprocess_with_avg_Glove
from .embeddings import preprocess_with_avg_fasttext

import pdb
classes = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'ship']
classes_index = [0, 1, 2, 3, 4, 5, 6]


class Reuters_Dataset(object):
    """Docstring for  Reuters_Dataset. """

    # original train dataset label counts
    # 'earn': 2843, 'acq': 1650, 'crude': 370, 'interest': 329, 'money-fx': 266, 'trade': 253, 'grain': 218,
    # 'corn': 157, 'dlr': 126, 'money-supply': 125, 'ship': 112, 'coffee': 101, 'sugar': 98, 'gold': 83,
    # 'gnp': 82, 'bop': 75, 'cpi': 67, 'cocoa': 55, 'carcass$: 49, 'oilseed': 49, 'copper': 45, 'reserves': 39,
    # 'jobs': 39, 'barley': 37, 'ipi': 37, 'alum': 35, 'iron-steel': 32, 'rubber': 31, $cotton': 28,
    # 'nat-gas': 26, 'palm-oil': 21, 'veg-oil': 19, 'meal-feed': 19, 'retail': 19, 'livestock': 18, 'tin': 17,
    # 'housing': 15, 'pet-chem': 15, 'orange': 14, 'wpi': 14, 'gas': 13, 'hog': 12, 'lei': 11, 'lumber': 10,
    # 'strategic-metal': 9, 'income': 8, 'zinc': 8$ 'fuel': 6, 'lead': 6, 'heat': 6, 'instal-debt': 5,
    # 'coconut': 4, 'silver': 4, 'soy-oil': 4, 'coconut-oil': 4, 'nickel': 3, 'l-cattl$': 3, 'cpu': 3,
    # 'rape-oil': 2, 'dmk': 2, 'groundnut': 2, 'tea': 2, 'potato': 2, 'jet': 2, 'castor-oil': 1, 'dfl': 1,
    # 'nzdlr': 1, 'co$ra-cake': 1, 'groundnut-oil': 1, 'soybean': 1, 'cotton-oil': 1, 'sun-oil': 1, 'rand': 1,
    # 'platinum': 1

    def __init__(self, root_dir: str):

        self._root_dir = root_dir
        self.classes = classes
        self.classes_index = classes_index
        self.train, self.test = reuters_dataset(root_dir)

        self.train_x = None
        self.train_y = None

        self.test_in_x = None
        self.test_in_y = None
        self.test_out_x = None
        self.test_out_y = None
        self.test_in_emb = None
        self.test_out_emb = None
        
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
     
   
    return in_data, in_labels, out_data, out_labels, label, c_in_labels    #,c_in_labels, c_out_labels


def reuters_dataset(directory='../data'):

    clean_txt = True
    train = True
    test = True

    nltk.download('reuters', download_dir=directory)
    if directory not in nltk.data.path:
        nltk.data.path.append(directory)

    doc_ids = reuters.fileids()

    ret = []
    splits = [
        split_set
        for (requested, split_set) in [(train, 'train'), (test, 'test')]
        if requested
    ]

    for split_set in splits:

        split_set_doc_ids = list(
            filter(lambda doc: doc.startswith(split_set), doc_ids))
        examples = []

        for id in split_set_doc_ids:
            if clean_txt:
                text = clean_text(reuters.raw(id))
            else:
                text = ' '.join(word_tokenize(reuters.raw(id)))
            labels = reuters.categories(id)

            examples.append({'text': text, 'label': labels})

        ret.append(Dataset(examples))

    ret_sentences = []
    ret_labels = []

    for ret_ in ret:

        sentence = []
        label = []

        for i, label_ in enumerate(ret_['label']):

            label_string = label_[0]
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
