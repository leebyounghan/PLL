#!/usr/bin/env python3
# -*- codeing: utf-8 -*-

import os

from .IMDB import IMDB_Dataset
from .reuters import Reuters_Dataset
from .news20 import news20_Dataset
from .trec import Trec_Dataset
from .agnews import agnews_Dataset
from .OOD import OOD_dataset
from .clinc_150 import clinc_150_Dataset
from .sst import sst_Dataset
from .rostd import ROSTD_Dataset
from .snips import snips_Dataset
from .bank import bank_Dataset
from .stack import stack_Dataset
from config import implemented_datasets, normal_class_index_list
import pdb

def load_dataset(dataset_name, data_path):

    assert dataset_name in implemented_datasets

    dataset = None
    if dataset_name == 'IMDB':
        # TODO: debug
        print("loading IMDB dataset...")
        IMDB_dataset = IMDB_Dataset(root_dir=data_path)
        print("preprocessing...")
        IMDB_dataset.preprocess_for_sentiment_understanding()
        dataset = IMDB_dataset.get_binary_labeled_data()

    elif dataset_name == 'reuters':
        print("loading reuters dataset...")
        reuters_dataset = Reuters_Dataset(root_dir=data_path)
        print("preprocessing...")
        reuters_dataset.preprocess_for_sentiment_understanding()
        dataset = reuters_dataset.get_binary_labeled_data()

    elif dataset_name == 'news20':
        print("loading news20 dataset...")
        news20_dataset = news20_Dataset(root_dir=data_path)
        print("preprocessing...")
        news20_dataset.preprocess_for_sentiment_understanding()
        dataset = news20_dataset.get_binary_labeled_data()

    elif dataset_name == 'agnews':
        print("loading agnews dataset...")
        news20_dataset = agnews_Dataset(root_dir=data_path)
        print("preprocessing...")
        news20_dataset.preprocess_for_sentiment_understanding()
        dataset = news20_dataset.get_binary_labeled_data()

    elif dataset_name == 'trec':
        print("loading trec dataset...")
        trec_dataset = Trec_Dataset(root_dir=data_path)
        print("preprocessing...")
        trec_dataset.preprocess_for_sentiment_understanding()
        dataset = trec_dataset.get_binary_labeled_data()

    elif dataset_name == "clinc":
        print("loading clinc_150 dataset...")
        dataset = clinc_150_Dataset()

    elif dataset_name == "sst":
        print("loading sst dataset...")
        dataset = sst_Dataset()

    elif dataset_name == 'rostd':
        print("loading rostd dataset ....")
        dataset = ROSTD_Dataset()

    elif dataset_name == 'snips':
        print('loading snipts dataset ...')
        dataset = snips_Dataset()

    elif dataset_name == 'banking':
        print('loading snipts dataset ...')
        dataset = bank_Dataset()

    elif dataset_name == 'stackoverflow':
        print('loading snipts dataset ...')
        dataset = stack_Dataset()



    return dataset

