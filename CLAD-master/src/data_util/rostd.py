#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
import numpy as np
import torch
import pdb
import os
import pdb


def ROSTD_Dataset():
    path = "/../../data/ROSTD/"
    cwd = os.getcwd()
    data_train = pd.read_csv(cwd + path + "train-en.tsv",
                             delimiter='\t',
                             header=None)
    test_in = pd.read_csv(cwd + path + "test-en.tsv",
                          delimiter='\t',
                          header=None)
    test_out = pd.read_csv(cwd + path + "OOD.tsv", delimiter='\t', header=None)
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
