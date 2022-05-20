# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import SGD, Adam
import torch.nn.utils as torch_utils
from typing import Optional
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans
from itertools import combinations
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import AdamW
from tqdm import tqdm
import config
import pdb

#torch.manual_seed(config.random_seed)
#torch.cuda.manual_seed(config.random_seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmanrk = False

log = config.logger


class testDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, length):
        self.encodings = encodings
        self.length = length

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        return item

    def __len__(self):
        return self.length


def set_text_predict(train_x, batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained(config.LM)
    train_encodings = tokenizer(train_x,
                                max_length=config.max_length,
                                truncation=True,
                                padding=True)
    num_data = len(train_encodings["input_ids"])
    data = testDataset(train_encodings, num_data)
    dataloader = DataLoader(data, batch_size=batch_size)
    return dataloader


def set_text_data(train_x, batch_size=8):

    tokenizer = AutoTokenizer.from_pretrained(config.LM)
    train_encodings = tokenizer(train_x,
                                truncation=True,
                                padding=True,
                                max_length=config.max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=True,
                                                    mlm_probability=0.10)
    num_data = len(train_encodings["input_ids"])
    data = testDataset(train_encodings, num_data)
    dataloader = DataLoader(data,
                            batch_size=batch_size,
                            collate_fn=data_collator)

    return dataloader, num_data


class Clustering_Module_bert():
    def __init__(self, train_x, test_in, test_out, n_components, batch_size=8):
        self.batch_size = batch_size
        self.n_components = n_components
        self.dataloader, self.num_data = set_text_data(train_x,
                                                       batch_size=batch_size)
        self.dataloader_predict = set_text_predict(train_x,
                                                   batch_size=batch_size)
        self.dataloader_in = set_text_predict(test_in, batch_size=batch_size)
        self.dataloader_out = set_text_predict(test_out, batch_size=batch_size)
        self.encoder = Encoder(n_components).to(config.device)
        self.cm = Cluster_Model(self.encoder)

    def train(self, epochs):
        self.cm = Cluster_Model(self.encoder)

        optimizer = AdamW(self.cm.parameters(),
                          lr=config.cluster_model_train_lr)

        data_iterator = tqdm(
            self.dataloader,
            leave='True',
            unit='batch',
            postfix={
                'epoch': -1,
                #  'acc': '%.4f' % 0.0,
                'loss': '%.6f' % 0.0,
                'dlb': '%.4f' % 0.0,
            })
        km = KMeans(n_clusters=self.n_components, n_init=30, n_jobs=-1)
        self.cm.train()
        self.cm.to(config.device)
        features = []

        for index, batch in enumerate(data_iterator):
            if ((isinstance(batch, tuple) or isinstance(batch, list))
                    and len(batch) == 2):
                batch, _ = batch

            #batch = batch.cuda(non_blocking=True)
            features.append(
                self.cm.encoder(batch)[1].detach().cpu())  #hiden features

        predicted = km.fit_predict(torch.cat(features).numpy())
        self.predicted_ = torch.tensor(predicted, dtype=torch.long)
        cluster_centers = torch.tensor(km.cluster_centers_,
                                       dtype=torch.float,
                                       requires_grad=True)
        #cluster_centers = cluster_centers.cuda(non_blocking=True)

        with torch.no_grad():
            self.cm.state_dict()['assignment.cluster_centers'].copy_(
                cluster_centers)
        loss_cls = nn.KLDivLoss(size_average=False)
        delta_label = None

        #train
        for epoch in range(epochs):
            features = []
            data_iterator = tqdm(self.dataloader,
                                 leave='True',
                                 unit='batch',
                                 postfix={
                                     'epoch': epoch,
                                     'loss': '%.8f' % 0.0,
                                     'dlb': '%.4f' % (delta_label or 0.0)
                                 })
            self.cm.train()
            for index, batch in enumerate(data_iterator):
                if ((isinstance(batch, tuple) or isinstance(batch, list))
                        and len(batch) == 2):
                    batch, label, _ = batch

                #batch = batch.cuda(non_blocking=True)

                label = batch['labels']
                cls, mlm, _ = self.cm(batch)

                target = target_distribution(cls).detach()
                #loss(cls, mlm)
                cls = loss_cls(cls.log(), target) / cls.shape[0]
                if config.cluster_loss_on:
                    loss = config.alpha * mlm + cls  # /self.batch_size

                else:
                    loss = mlm

                data_iterator.set_postfix(epoch=epoch,
                                          loss='%.8f' % float(loss.item()),
                                          dlb='%.4f' % (delta_label or 0.0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(closure=None)
                features.append(self.cm.encoder(batch)[1].detach().cpu())
                if index % 10 == 0:  # update_freq = 10
                    loss_value = float(loss.item())
                    data_iterator.set_postfix(
                        epoch=epoch,
                        loss='%.8f' % loss_value,
                        dlb='%.4f' % (delta_label or 0.0),
                    )

        print("training dec ended.")

    def predict(self):
        features = []
        actual = []
        embed = []
        self.cm.eval()
        #  for batch in data_iterator:

        for batch in self.dataloader_predict:
            if ((isinstance(batch, tuple) or isinstance(batch, list))
                    and len(batch) == 2):
                batch, value = batch

            #batch = batch.cuda(non_blocking=True)
            cluster, _, emb = self.cm.predict(batch)
            cluster = cluster.detach().cpu()
            emb = emb.detach().cpu()
            embed.append(emb)
            features.append(cluster)

        #print(torch.cat(features).max(1))
        return torch.cat(features).max(1)[1], torch.cat(embed), self.predicted_

    def predict_out(self):
        features_in = []
        features_out = []
        embed_in = []
        embed_out = []
        self.cm.eval()
        #  for batch in data_iterator:

        for batch in self.dataloader_in:
            if ((isinstance(batch, tuple) or isinstance(batch, list))
                    and len(batch) == 2):
                batch, value = batch

            #batch = batch.cuda(non_blocking=True)
            cluster, _, emb = self.cm.predict(batch)
            cluster = cluster.detach().cpu()
            emb = emb.detach().cpu()
            embed_in.append(emb)
            features_in.append(cluster)

        for batch in self.dataloader_out:
            if ((isinstance(batch, tuple) or isinstance(batch, list))
                    and len(batch) == 2):
                batch, value = batch

            #batch = batch.cuda(non_blocking=True)
            cluster, _, emb = self.cm.predict(batch)
            cluster = cluster.detach().cpu()
            emb = emb.detach().cpu()
            embed_out.append(emb)
            features_out.append(cluster)

        #print(torch.cat(features).max(1))
        return torch.cat(features_in).max(1)[1], torch.cat(
            embed_in), torch.cat(features_out).max(1)[1], torch.cat(embed_out)


class Cluster_Model(nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super(Cluster_Model, self).__init__()
        self.encoder = encoder
        self.assignment = ClusterAssignment(
            self.encoder.n_components, 768, alpha=1.0
        )  # alpha represent the degrees of freedom in the t-distribution

    def forward(self, x):
        out = self.assignment(self.encoder(x))
        return out

    def predict(self, x):
        with torch.no_grad():
            out = self.assignment(self.encoder.predict(x))
        return out

    def encode(self, x):
        with torch.no_grad():
            out = self.encoder(x)
        return out


class ClusterAssignment(nn.Module):
    def __init__(self,
                 cluster_number: int,
                 embedding_dimension: int,
                 alpha: float = 1.0,
                 cluster_centers: Optional[torch.Tensor] = None) -> None:
        super(ClusterAssignment, self).__init__()
        self.cluster_number = cluster_number
        self.embedding_dimension = embedding_dimension
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.cluster_number,
                                                  self.embedding_dimension,
                                                  dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    #  compute the soft assignment for a batch of feature vectors, returning a batch of assignments
    #  for each cluster.
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        norm_squared = torch.sum(
            (batch[1].unsqueeze(1) - self.cluster_centers)**2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power

        return numerator / torch.sum(numerator, dim=1,
                                     keepdim=True), batch[0], batch[1]


class Encoder(nn.Module):
    def __init__(self, n_components):
        super(Encoder, self).__init__()
        self.encoder = AutoModelForMaskedLM.from_pretrained(config.LM)
        # self.encoder.bert.embeddings.requires_grad_(False)
        self.n_components = n_components

    def forward(self, x):

        input_ids = x['input_ids'].to(config.device)
        attention_mask = x["attention_mask"].to(config.device)
        labels = x["labels"].to(config.device)
        #pdb.set_trace()
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels,
                               output_hidden_states=True)
        loss = outputs[0]  #mlm
        hiden_state = mean_pooling(outputs.hidden_states[-1], attention_mask)

        return loss, hiden_state

    def predict(self, x):
        input_ids = x['input_ids'].to(config.device)
        attention_mask = x["attention_mask"].to(config.device)
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               output_hidden_states=True)
        hiden_state = outputs.hidden_states[-1]
        avg_pool = mean_pooling(hiden_state, attention_mask)

        return None, avg_pool


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch**2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output  #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


def max_pooling(model_output, attention_mask):
    token_embeddings = model_output.clone(
    )  #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    token_embeddings[input_mask_expanded ==
                     0] += -1e8  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]
