#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from torchnlp.datasets.dataset import Dataset
from torchnlp.datasets import imdb_dataset


from nltk import word_tokenize
from .misc import clean_text

import nltk

import config
from .embeddings import preprocess_with_avg_bert
from .embeddings import preprocess_with_s_bert
from .embeddings import preprocess_with_avg_Glove
from .embeddings import preprocess_with_avg_fasttext

import pdb



class IMDB_Dataset(object):
    """Docstring for  Reuters_Dataset. """

  


    def __init__(self, root_dir: str):

        self._root_dir = root_dir

        self.train, self.test = IMDB_dataset(root_dir, False)

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

        if config.classifier_type == 'bert':
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





def IMDB_dataset(root_dir,clen = False):
        
    train, test = imdb_dataset(directory=root_dir, train=True, test=True)
    a = {'label':[], 'sentence': []}
    for i in train:
        if i["sentiment"] == "pos":
            a['label'].append(0)
            if clen == True:
                a["sentence"].append(clean_text(i["text"].lower()))
            else:
                a["sentence"].append(i["text"].lower()) 
        
        else:
            a['label'].append(1)
            if clen == True:
                a["sentence"].append(clean_text(i["text"].lower()))
            else:
                a["sentence"].append(i["text"].lower())          
            
    b = {'label':[], 'sentence': []}        
    for i in test:
        if i["sentiment"] == "pos":
            b['label'].append(0)
            if clen == True:
                b["sentence"].append(clean_text(i["text"].lower()))
            else:
                b["sentence"].append(i["text"].lower())
        
        else:
            b['label'].append(1)
            if clen == True:
                b["sentence"].append(clean_text(i["text"].lower()))
            else:
                b["sentence"].append(i["text"].lower())
                
            
    train = pd.DataFrame(a)
    test = pd.DataFrame(b)
    
    return train, test
            