#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import json
import numpy as np
import torch
import pdb


def load_dataset(data_name, data_type="full"):
    with open("../../data/data_full.json", 'r') as f:
        data = json.load(f)

    field = "_".join(data_name.split("_")[1:])
    dataset = data[field]
    data_df = pd.DataFrame(dataset, columns=["sentance", "labels"])
    data_df["labels"] = data_df['labels'].astype("category").cat.codes
    return data_df


def clinc_150_Dataset():
    data_train = load_dataset("clinc150_train")
    test_in = load_dataset("clinc150_test")
    test_out = load_dataset("clinc150_oos_test")

    dataset = {
        "train_x": data_train["sentance"].tolist(),
        "train_y": torch.tensor(data_train["labels"].tolist()),
        "test_in": test_in["sentance"].tolist(),
        "test_out": test_out["sentance"].tolist(),
        "test_in_emb": test_in["sentance"].tolist(),
        "test_out_emb": test_out["sentance"].tolist(),
        "train_text": data_train["sentance"].tolist()
    }
    return dataset
