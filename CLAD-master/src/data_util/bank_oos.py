#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
import numpy as np
import torch
import pandas as pd
import os
import pdb



def load_intent_examples(file_path, do_lower_case=True):
    examples = []

    with open('{}/seq.in'.format(file_path), 'r', encoding="utf-8") as f_text, open('{}/label'.format(file_path), 'r', encoding="utf-8") as f_label:
        for text, label in zip(f_text, f_label):
            e = (text.strip(), label.strip())
            examples.append(e)
        
        df = pd.DataFrame(examples, columns = ['text', 'labels'])
        
    return df


def bank_Dataset(normal_num):
    cwd = os.getcwd()
    path ='/../../data/BANKING77-OOS'
    print('bank-oos')
    data_train = load_intent_examples(cwd + path + "/train")
    data_val = load_intent_examples(cwd + path + "/valid")

    test_in = load_intent_examples(cwd + path + '/test')
    
    if normal_num[0] == 0 :
        test_out = load_intent_examples(cwd + path + '/id-oos/test')
    else :
        test_out = load_intent_examples(cwd + path + '/ood-oos/test')
   

    dataset = {
        "train_x":
        data_train['text'].tolist(),
        "train_y":
        torch.tensor(data_train['labels'].astype('category').cat.codes.tolist()),
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





