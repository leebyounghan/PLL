#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch.autograd import Variable
from transformers import AutoModelForMaskedLM, AutoModel, BertConfig

import numpy as np

import config

import pdb

# -------------------------
# Neural-Network-based classifier
# -------------------------


# for linear classifier
class Linear_Model(nn.Module):
    def __init__(self, input_dim):
        super(Linear_Model).__init__()
        self.out_features_dim = config.cluster_num
        self.linear = nn.Linear(input_dim, self.out_features_dim)

    def forward(self, x):
        if (len(x.shape) >= 4):
            x = torch.reshape(x, (len(x), -1))
        elif (len(x.shape) >= 3):
            x = torch.reshape(x, (1, -1))
        return self.linear(x)

    def predict(self, x):
        if (len(x.shape) >= 4):
            x = torch.reshape(x, (len(x), -1))
        elif (len(x.shape) >= 3):
            x = torch.reshape(x, (1, -1))
        dataloader = DataLoader(x, batch_size=128)
        predicted = []
        with torch.no_grad():
            for _, data in enumerate(dataloader):
                out_features = self.linear(data)
                predict_sm = F.softmax(out_features)
                predict_sm = predict_sm.detach().cpu().numpy()
                for i in range(len(predict_sm)):
                    predicted.append(
                        np.where(predict_sm[i] == max(predict_sm[i]))[0][0])
        return predicted


class FC3_Model(nn.Module):
    def __init__(self, input_dim):
        super(FC3_Model, self).__init__()
        self.out_features_dim = config.cluster_num
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, self.out_features_dim)

    def forward(self, x):
        output1 = F.relu(self.linear1(x))
        output2 = F.relu(self.linear2(output1))
        output3 = self.linear3(output2)
        return output3

    def predict(self, x):
        dataloader = DataLoader(x, batch_size=128)
        predicted = []
        with torch.no_grad():
            for _, data in enumerate(dataloader):
                out_features1 = F.relu(self.linear1(data))
                out_features2 = F.relu(self.linear2(out_features1))
                out_features3 = self.linear3(out_features2)
                predict_sm = F.softmax(out_features3)
                predict_sm = predict_sm.detach().cpu().numpy()
                for i in range(len(predict_sm)):
                    predicted.append(
                        np.where(predict_sm[i] == max(predict_sm[i]))[0][0])
            return predicted


class Bert_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_features_dim = config.cluster_num
        self.bert = AutoModel.from_pretrained(config.LM)
        self.embdding = self.bert.get_input_embeddings()
        self.linear = nn.Linear(768, self.out_features_dim)

    def forward(self,
                input_ids,
                attention_mask,
                odin,
                ge,
                labels=None):  #input
        if odin == 1:
            inputs = Variable(self.embdding(input_ids), requires_grad=True)
            if ge == None:
                tempInputs = inputs
            else:
                tempInputs = torch.add(inputs.data, 0.0001, ge)
            outputs = self.bert(inputs_embeds=tempInputs)
            logits = self.linear(outputs.pooler_output)
            return logits, inputs, outputs.pooler_output  #inputs for gradient
        else:
            outputs = self.bert(input_ids)
            pooled_output = pooled = outputs.pooler_output
            logits = self.linear(pooled_output)
            #  dist = ((pooled.unsqueeze(1) - pooled.unsqueeze(0))**2).mean(-1)
            #  mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
            #  mask = mask - torch.diag(torch.diag(mask))
            #  neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
            #  max_dist = (dist * mask).max()
            #  cos_loss = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) + (
            #  F.relu(max_dist - dist) *
            #  neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-3)
            #  cos_loss = cos_loss.mean()
            #
            return logits  #, cos_loss  #output

    def predict(self, x):
        train_pred = []
        eval_dataloader = DataLoader(x, batch_size=4)
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                outputs = self.bert(input_ids, attention_mask)
                logits = F.softmax(self.linear(outputs.pooler_output))
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                train_pred.extend(pred)

        return train_pred


class Bert_Model_mlm(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.out_features_dim = config.cluster_num
        self.bert = model
        self.dropout = nn.Dropout(p=0.2)
        #self.bert.bert.embeddings.requires_grad_(False)#freeze
        self.linear = nn.Linear(768, self.out_features_dim)
        self.activation = torch.tanh

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask,
                            output_hidden_states=True,
                            output_attentions=True)

        if config.pooling == "max":
            pooled_output = hidden_state = max_pooling(
                outputs.hidden_states[-1], attention_mask)
        elif config.pooling == "mean":
            pooled_output = hidden_state = mean_pooling(
                outputs.hidden_states[-1], attention_mask)
        elif config.pooling == "cls":
            pooled_output = hidden_state = self.activation(
                outputs.hidden_states[-1][:, 0, :])

            pooled_output = self.activation(self.dropout(pooled_output))

        #  if odin == 1:
            #  inputs = Variable(hiden_state, requires_grad=True)
            #  if ge == None:
                #  tempInputs = inputs
            #  else:
                #  tempInputs = torch.add(hiden_state.data, -config.perturbation,
                                       #  ge)
#
            #  logits = self.linear(tempInputs)
            #  return logits, inputs, hiden_state  #inputs for gradient
        #  else:
            #  logits = self.linear(pooled_output)
        logits = self.linear(pooled_output)
        return logits, hidden_state  #,cos_loss #output

    def predict(self, x):
        train_pred = []
        eval_dataloader = DataLoader(x, batch_size=4)
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                outputs = self.bert(input_ids,
                                    attention_mask,
                                    output_hidden_states=True)
                hiden_state = self.activation(outputs.hidden_states[-1][:,
                                                                        0, :])
                logits = F.softmax(self.linear(hiden_state))
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                train_pred.extend(pred)

        return train_pred


class Bert_Model_scr(nn.Module):
    def __init__(self):
        super().__init__()
        print("bert_src")
        self.out_features_dim = config.cluster_num
        self.bert = AutoModelForMaskedLM(BertConfig(vocab_size=30522))
        self.linear = nn.Linear(768, self.out_features_dim)

    def forward(self, input_ids, attention_mask, odin, ge):
        outputs = self.bert(input_ids, attention_mask)
        if odin == 1:
            inputs = Variable(outputs.pooler_output, requires_grad=True)
            if ge == 0:
                tempInputs = inputs
            else:
                tempInputs = torch.add(inputs.data, -config.perturbation, ge)

            logits = self.linear(tempInputs)
            return logits, inputs, outputs.pooler_output  #inputs for gradient
        else:
            logits = self.linear(outputs.pooler_output)
            return logits  #output

    def predict(self, x):
        train_pred = []
        eval_dataloader = DataLoader(x, batch_size=4)
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                outputs = self.bert(input_ids, attention_mask)
                logits = F.softmax(self.linear(outputs.pooler_output))
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                train_pred.extend(pred)

        return train_pred


class Bert_Model_cls(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.out_features_dim = config.cluster_num
        self.bert = model

        #self.bert.bert.embeddings.requires_grad_(False)#freeze

        self.linear = nn.Linear(768, self.out_features_dim)
        self.activation = torch.tanh

    def forward(self, input_ids, attention_mask, odin, ge, labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            output_hidden_states=True)
        loss_mlm = outputs[0]
        hiden_state = outputs.hidden_states[-1].mean(1)  #for cheak mlm
        #hiden_state = self.activation(outputs.hidden_states[-1][:,0,:]) # for cls

        if odin == 1:
            inputs = Variable(hiden_state, requires_grad=True)
            if ge == None:
                tempInputs = inputs
            else:
                tempInputs = torch.add(hiden_state, -config.perturbation, ge)

            logits = self.linear(tempInputs)
            return logits, inputs, hiden_state  #inputs for gradient
        else:
            logits = self.linear(hiden_state)
            return logits, loss_mlm  #output

    def predict(self, x):
        train_pred = []
        eval_dataloader = DataLoader(x, batch_size=4)
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                outputs = self.bert(input_ids,
                                    attention_mask,
                                    output_hidden_states=True)
                hiden_state = self.activation(outputs.hidden_states[-1][:,
                                                                        0, :])
                logits = F.softmax(self.linear(hiden_state))
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                train_pred.extend(pred)

        return train_pred


def max_pooling(model_output, attention_mask):
    token_embeddings = model_output  #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    token_embeddings[input_mask_expanded ==
                     0] = -1e8  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output  #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)
