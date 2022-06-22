#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
import numpy as np
import torch
import pdb

seed = 42


def sst_Dataset():
    train_df = load_extra_dataset("../../data/sst/sst-train.txt", label=1)
    test_df = load_extra_dataset("../../data/sst/sst-test.txt", label=1)
    ood_snli_df = load_extra_dataset("../../data/sst/snli-dev.txt",
                                    drop_index=True,
                                     label=0)
    ood_rte_df = load_extra_dataset("../../data/sst/rte-dev.txt",
                                    drop_index=True,
                                    label=0)
    ood_20ng_df = load_extra_dataset("../../data/sst/20ng-test.txt",
                                     drop_index=True,
                                     label=0)
    ood_multi30k_df = load_extra_dataset("../../data/sst/multi30k-val.txt",
                                         drop_index=True,
                                         label=0)
    ood_snli_df = ood_snli_df  #.sample(n=500, random_state=seed)
    ood_rte_df = ood_rte_df  #.sample(n=500, random_state=seed)
    ood_20ng_df = ood_20ng_df  #.sample(n=500, random_state=seed)
    ood_multi30k_df = ood_multi30k_df  #.sample(n=500, random_state=seed)
    ood_df = pd.concat([ood_snli_df, ood_rte_df, ood_20ng_df, ood_multi30k_df])
    # ood_df = ood_df.sample(n=len(test_df), random_state=seed)
    # pdb.set_trace()
    dataset = {
        "train_x": train_df["sentence"].tolist(),
        "train_y": torch.tensor(train_df["labels"].tolist()),
        "test_in": test_df["sentence"].tolist(),
        "test_out": ood_df["sentence"].tolist(),
        "test_in_emb": test_df["sentence"].tolist(),
        "test_out_emb": ood_df["sentence"].tolist(),
        "train_text": train_df["sentence"].tolist()
    }
    return dataset


def load_extra_dataset(file_path="../../data/SSTSentences.txt",
                       drop_index=False,
                       label=0):
    df = pd.read_csv(file_path, sep='\t', header=0)
    df['labels'] = label
    df.rename(columns={'text': 'sentence'}, inplace=True)
    if drop_index:
        df.drop(columns='index', inplace=True)
    df.dropna(inplace=True)
    return df

def ood_ood():
    ood_snli_df = load_extra_dataset("../../data/sst/snli-dev.txt",
                                    drop_index=True,
                                     label=0)
    ood_rte_df = load_extra_dataset("../../data/sst/rte-dev.txt",
                                    drop_index=True,
                                    label=0)
    ood_20ng_df = load_extra_dataset("../../data/sst/20ng-test.txt",
                                     drop_index=True,
                                     label=0)
    ood_multi30k_df = load_extra_dataset("../../data/sst/multi30k-val.txt",
                                         drop_index=True,
                                         label=0)
    ood_snli_df = ood_snli_df.sample(n=500, random_state=seed)
    ood_rte_df = ood_rte_df.sample(n=500, random_state=seed)
    ood_20ng_df = ood_20ng_df.sample(n=500, random_state=seed)
    ood_multi30k_df = ood_multi30k_df.sample(n=500, random_state=seed)
    ood_df = pd.concat([ood_snli_df, ood_rte_df, ood_20ng_df, ood_multi30k_df])
    
    return ood_df['sentence'].tolist()
