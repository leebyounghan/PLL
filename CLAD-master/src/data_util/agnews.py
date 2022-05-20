#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch

from torchnlp.datasets.dataset import Dataset
import nltk

nltk.download('stopwords')
nltk.download('punkt')
import os
from nltk import word_tokenize
from .misc import clean_text

import config
from .embeddings import preprocess_with_avg_bert
from .embeddings import preprocess_with_s_bert
from .embeddings import preprocess_with_avg_Glove
from .embeddings import preprocess_with_avg_fasttext
import pdb


class agnews_Dataset(object):
    """Docstring for  Reuters_Dataset. """
    def __init__(self, root_dir: str):

        self._root_dir = root_dir
        self.train, self.test = agnews_dataset(root_dir)

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
            pass  #no emb
        return self.train_x, self.train_y, self.test_x, self.test_y, self.train_text, self.test_text

    # get label 0 for normal class and 1 for the rest
    def get_binary_labeled_data(self):
        #TODO: change the normal class index with configuration

        self.train_x, _, _, _, label, self.train_y = divide_data_label(
            self.train_x, self.train_y, train=True)

        if config.classifier_type in ['bert', 'DEC_bert', "bert_scr"]:
            print("we in bert")
            self.train_text, _, _, _, _, c_in_label = divide_data_label(
                self.train_text, label, train=True)

            self.test_in_x, self.test_in_y, self.test_out_x, self.test_out_y, _, _ = divide_data_label(
                self.test_text, self.test_y, train=False)
            self.test_in_emb, _, self.test_out_emb, _, _, _ = divide_data_label(
                self.test_x, self.test_y, train=False)
            self.test_in_emb = torch.tensor(self.test_in_emb)
            self.test_out_emb = torch.tensor(self.test_out_emb)
        else:
            self.test_in_x, self.test_in_y, self.test_out_x, self.test_out_y, _, _ = divide_data_label(
                self.test_x, self.test_y, train=False)

            self.test_in_x = torch.tensor(self.test_in_x)
            self.test_out_x = torch.tensor(self.test_out_x)

        self.train_x = torch.tensor(self.train_x)
        self.train_y = torch.tensor(self.train_y)
        dataset = {
            "train_x": self.train_x,
            "train_y": self.train_y,
            "test_in": self.test_in_x,
            "test_out": self.test_out_x,
            "test_in_emb": self.test_in_emb,
            "test_out_emb": self.test_out_emb,
            "train_text": self.train_text,
        }

        return dataset


def divide_data_label(dataset, label, train=False):
    in_data = []
    out_data = []
    in_labels = []
    out_labels = []
    c_out_labels = []
    c_in_labels = []
    c_in_labels_check = []
    for i, _d in enumerate(dataset):
        data_x = _d
        data_y = label[i]
        if (data_y in config.normal_class_index_list):
            in_data.append(data_x)
            in_labels.append(0)
            c_in_labels.append(config.normal_class_index_list.index(data_y))
            c_in_labels_check.append(data_y)
        else:
            if (train): continue
            else:
                out_data.append(data_x)
                out_labels.append(1)
                c_out_labels.append(data_y)  # for data chek

    return in_data, in_labels, out_data, out_labels, label, c_in_labels


def agnews_dataset(directory='../data',
                   train=True,
                   test=True,
                   clean_text=False):
    """
    Load the AG News dataset.

    Args:
        directroy (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        test (bool, optional): If to load the test split of the dataset.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train and test) depending on if their respective boolean argument
        is ``True``.
    """

    if directory not in nltk.data.path:
        nltk.data.path.append(directory)

    ret = []
    splits = [
        split_set
        for (requested, split_set) in [(train, 'train'), (test, 'test')]
        if requested
    ]
    for split_set in splits:
        # AGNews dataset downloaded from kaggle : https://www.kaggle.com/amananandrai/ag-news-classification-dataset/version/2
        data_path = os.path.join(
            *[os.getcwd(), directory, 'agnews', split_set]) + '.csv'
        dataset = pd.read_csv(data_path,
                              thousands=',',
                              index_col=0,
                              encoding='utf-8')
        dataset.head()
        examples = []
        for id in range(len(dataset)):
            label = int(dataset.index[id])
            # title = dataset.iloc[id]['Title']
            text = dataset.iloc[id]['Title'] + ' ' + dataset.iloc[id][
                'Description']
            if clean_text:
                text = clean_text(text)
            else:
                text = text
            if text:
                examples.append({'sentence': text, 'label': label})

        ret.append(Dataset(examples))

    ret_sentences = []
    ret_labels = []

    for ret_ in ret:
        ret_sentences.append(ret_['sentence'])
        ret_labels.append(ret_['label'])

    train = pd.DataFrame({
        'sentence': ret_sentences[0],
        'label': ret_labels[0]
    })
    #  train, val = np.split(train.sample(frac=1), [int(
    #      (0.9) * len(train))])  # split 9:1 = train:val
    test = pd.DataFrame({'sentence': ret_sentences[1], 'label': ret_labels[1]})

    return train, test
