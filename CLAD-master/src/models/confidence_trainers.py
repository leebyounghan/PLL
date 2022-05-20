#!/usr/bin/env python3
# -*- codeing: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import numpy as np
import time
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
from models.classifiers import Linear_Model, FC3_Model, Bert_Model, Bert_Model_mlm, Bert_Model_scr, Bert_Model_cls

from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import InputExample
from tqdm import tqdm

import config

import pdb

# -------------------------
# Naive ML-based classifier
# -------------------------


# for basic test
def Linear_classifier(train_data, train_cluster, n_epochs, lr):

    _, input_size = train_data.shape

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    #  test_data = scaler.fit_transform(test_data)

    train_data = torch.from_numpy(train_data.astype(np.float32)).cuda(
        config.device)
    train_cluster = torch.from_numpy(train_cluster).cuda(config.device)

    model = Linear_Model(input_dim=input_size)
    model = nn.DataParallel(model).cuda(config.device)
    criterion = nn.CrossEntropyLoss()  # Log Softmax + ClassNLL Loss
    optimizer = Adam(model.parameters(), lr=lr)

    for iter_ in range(n_epochs):
        outputs = model(train_data)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, train_cluster)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # TODO: loss can be ploted
        #  train_losses[iter_] = loss.item()
        if (iter_ + 1) % 10 == 0:
            print("Epoch {}/{}, Training loss: {}".format(
                (iter_ + 1), n_epochs, loss.item()))
    return model


def FC3_classifier(train_data, train_cluster, n_epochs, lr):
    if (len(train_data) > 2):
        train_data = torch.reshape(train_data, (len(train_data), -1))

    _, input_size = train_data.shape
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = torch.from_numpy(train_data.astype(np.float32)).cuda("cuda")
    train_cluster = train_cluster.cuda("cuda")
    model = FC3_Model(input_dim=input_size).to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    train_losses = np.zeros(n_epochs)

    for iter_ in range(n_epochs):
        outputs = model(train_data)
        outputs = torch.squeeze(outputs)  #shape이 이상하다
        loss = criterion(outputs, train_cluster)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iter_ + 1) % 10 == 0:
            print("Epoch {}/{}, Training loss: {}".format(
                (iter_ + 1), n_epochs, loss.item()))
    return model


#need to bulid
def BERT(train_data, n_epochs, lr):

    #basic

    train_loader = DataLoader(train_data, batch_size=6, shuffle=True)
    #pdb.set_trace()
    if config.classifier_type == "bert_scr":
        model = Bert_Model_scr().to(config.device)
    else:
        model = Bert_Model().to(config.device)

    model.train()
    criterion_cls = nn.CrossEntropyLoss()
    optim = AdamW(model.parameters(), lr=config.classifier_lr)

    for epoch in range(config.classifier_epochs):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            outputs = model(input_ids, attention_mask, 0, 0,
                            labels=labels)  #input_ids,attention_mask, odin, ge
            loss_cls = criterion_cls(outputs, labels)
            loss = loss_cls

            loss_val = loss.item()

            loss.backward()
            optim.step()

        print("Epoch {}/{}, Training loss: {}".format((epoch + 1), n_epochs,
                                                      loss_cls.item()))

    return model


def BERT_mlm(train_data, n_epochs, lr, models, batch):

    #basic

    train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
    model = Bert_Model_mlm(models).to(config.device)
    model.train()
    criterion_cls = nn.CrossEntropyLoss()
    optim = AdamW(model.parameters(), lr=config.classifier_lr)
    loss_val = 0.0
    for epoch in range(config.classifier_epochs):
        for i, batch in enumerate(train_loader, 0):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            outputs = model(input_ids, attention_mask, 0, 0,
                            labels=labels)  #input_ids,attention_mask, odin, ge
            loss_cls = criterion_cls(outputs, labels)
            loss = loss_cls

            loss_val += loss.item()

            loss.backward()
            optim.step()
            if i % 100 == 99:
                print("Epoch {}/{}, Training loss: {}".format((epoch + 1), i,
                                                              loss_val / 100))
                loss_val = 0.0

    return model


def BERT_cls(train_data, n_epochs, lr, models):

    #basic

    model = Bert_Model_cls(models).to(config.device)
    model.train()
    criterion_cls = nn.CrossEntropyLoss()
    optim = AdamW(model.parameters(), lr=config.classifier_lr)
    loss_val = 0.0
    for epoch in range(config.classifier_epochs):
        for i, batch in enumerate(train_data, 0):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            cls_label = batch['cls'].to(config.device)
            mlm_label = batch['labels'].to(config.device)
            outputs, mlm_loss = model(
                input_ids, attention_mask, 0, 0,
                mlm_label)  #input_ids,attention_mask, odin, ge
            loss_cls = criterion_cls(outputs, cls_label)
            loss = loss_cls + mlm_loss

            loss_val += loss_cls.item()

            loss.backward()
            optim.step()
            if i % 100 == 99:
                print("Epoch {}/{}, Training loss: {}".format((epoch + 1), i,
                                                              loss_val / 100))
                loss_val = 0.0

    return model
