#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
import numpy as np
import torch
import pdb
import os
import pdb


def snips_Dataset(SEED):
    path = "/../../data/sinps/seed_" + str(SEED)
    cwd = os.getcwd()
    data_train = pd.read_csv(cwd + path + "/OODRemovedtrain.tsv",
                             delimiter='\t',
                             header=None)
    test = pd.read_csv(cwd + path + '/test.tsv', delimiter='\t', header=None)
    test_in = test[test[0] != 'outOfDomain']
    test_out = test[test[0] == 'outOfDomain']

    dataset = {
        "train_x":
        data_train[2].tolist(),
        "train_y":
        torch.tensor(data_train[0].astype('category').cat.codes.tolist()),
        "test_in":
        test_in[2].tolist(),
        "test_out":
        test_out[2].tolist(),
        "test_in_emb":
        test_in[2].tolist(),
        "test_out_emb":
        test_out[2].tolist(),
        "train_text":
        data_train[2].tolist()
    }

    return dataset
