#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
import numpy as np
import torch
import pdb
import os
import pdb
import config

def bank_Dataset():
    path = "/../../data/banking/seed_" + str(config.normal_class_index_list[0])
    print(path)
    cwd = os.getcwd()
    data_train = pd.read_csv(cwd + path + "/train.csv")
    test = pd.read_csv(cwd + path + '/test.csv')
    test_in = test[test['label'] != 'oos']
    test_out = test[test['label'] == 'oos']
    
    dataset = {
        "train_x":
        data_train['text'].tolist(),
        "train_y":
        torch.tensor(data_train['label'].astype('category').cat.codes.tolist()),
        "test_in":
        test_in['text'].tolist(),
        "test_out":
        test_out['text'].tolist(),
        "test_in_emb":
        test_in['text'].tolist(),
        "test_out_emb":
        test_out['text'].tolist(),
        "train_text":
        data_train['text'].tolist()
    }

    return dataset
