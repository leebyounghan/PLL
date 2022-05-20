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
from transformers import BertForMaskedLM, BertConfig, BertTokenizer,DataCollatorForLanguageModeling
from transformers import AdamW
from tqdm import tqdm
import config
import pdb

<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
log = config.logger

class testDataset(torch.utils.data.Dataset):
    def __init__(self, encodings,length):
        self.encodings = encodings
        self.length = length
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.length
        
        
def set_text_predict(train_x, batch_size = 8):
    
    tokenizer = BertTokenizer("/home/byounghan96/workspace/NCLAD-tset/CLAD-master/src/comp-vocab.txt")
    train_encodings = tokenizer(train_x, truncation=True, max_length = 512, padding = True) 
    train_encodings['token_len'] = [ len(np.nonzero(i)[0])  for i in train_encodings["input_ids"]]
    num_data = len(train_encodings["input_ids"])
    data = testDataset(train_encodings, num_data) 
    dataloader = DataLoader(data, batch_size = batch_size)
    
    
    return dataloader

def set_text_predict_cls(train_x,labels,batch_size = 8):
    
    tokenizer = BertTokenizer("/home/byounghan96/workspace/NCLAD-tset/CLAD-master/src/comp-vocab.txt")
    train_encodings = tokenizer(train_x, truncation=True, max_length = 512, padding = True) 
    train_encodings['token_len'] = [ len(np.nonzero(i)[0])  for i in train_encodings["input_ids"]]
    train_encodings["labels"] = labels
    num_data = len(train_encodings["input_ids"])
    data = testDataset(train_encodings, num_data) 
    dataloader = DataLoader(data, batch_size = batch_size)
 

def mlm_cls_dataset(train_x, cls, batch_size = 8):
    
    tokenizer = BertTokenizer("/home/byounghan96/workspace/NCLAD-tset/CLAD-master/src/comp-vocab.txt")
    train_encodings = tokenizer(train_x, truncation=True, max_length = 512,padding =True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.10)
    train_encodings['token_len'] = [ len(np.nonzero(i)[0])  for i in train_encodings["input_ids"]]
    train_encodings["cls"] = cls
    num_data = len(train_encodings["input_ids"])
    data = testDataset(train_encodings, num_data)
    dataloader = DataLoader(data, batch_size = batch_size, collate_fn=data_collator)
    
    return dataloader, num_data



def set_text_data(train_x, batch_size = 8): 
    tokenizer = BertTokenizer("/home/byounghan96/workspace/NCLAD-tset/CLAD-master/src/comp-vocab.txt")
    train_encodings = tokenizer(train_x, truncation=True, max_length = 512,padding =True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.10)
    train_encodings['token_len'] = [ len(np.nonzero(i)[0])  for i in train_encodings["input_ids"]]
    num_data = len(train_encodings["input_ids"])
    data = testDataset(train_encodings, num_data)
    dataloader = DataLoader(data, batch_size = batch_size, collate_fn=data_collator)
    
    return dataloader, num_data






class Clustering_Module_bert_scr():
    def __init__(self, train_x, test_out, n_components, batch_size = 8):
        self.batch_size = batch_size
        self.n_components = n_components
        self.dataloader, self.num_data = set_text_data(train_x, batch_size=batch_size)
        self.dataloader_predict = set_text_predict(train_x, batch_size=batch_size) 
        self.dataloader_out, _ =set_text_data(test_out,batch_size=batch_size)
        self.encoder = Encoder(n_components).to(config.device)
        self.cm = Cluster_Model(self.encoder)




    def train(self, epochs):
        self.cm = Cluster_Model(self.encoder)
       
        optimizer = AdamW(self.cm.parameters(), lr=3e-5) 

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
        km = KMeans(n_clusters=self.n_components,
                    n_init=10,
                    n_jobs=-1)
        self.cm.train()
        self.cm.to(config.device)
        features = []
        actual = []


        for index, batch in enumerate(data_iterator):
            if ((isinstance(batch, tuple) or isinstance(batch, list))
                    and len(batch) == 2):
                batch, _ = batch

                
            #batch = batch.cuda(non_blocking=True)           
            features.append(self.cm.encoder(batch)[1].detach().cpu())  #hiden features
        
        predicted = km.fit_predict(torch.cat(features).numpy())
        predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)

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
                    batch, label, _ = batch #
                
                #batch = batch.cuda(non_blocking=True)

                label = batch['labels']
                cls, mlm, _ = self.cm(batch)

                target = target_distribution(cls).detach()
                #loss(cls, mlm)
                cls = loss_cls(cls.log(), target) / cls.shape[0]
                loss = cls + mlm/self.batch_size
                
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
        return torch.cat(features).max(1)[1], torch.cat(embed) 

    def predict_out(self):
        features = []
        actual = []
        embed = []
        self.cm.eval()
        #  for batch in data_iterator:
            
        for batch in self.dataloader_out:
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
        return torch.cat(features).max(1)[1], torch.cat(embed) 


        
class Cluster_Model(nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super(Cluster_Model, self).__init__()
        self.encoder = encoder
        self.assignment = ClusterAssignment(
            self.encoder.n_components,
            768,
            alpha=1.0
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
        
        return numerator / torch.sum(numerator, dim=1, keepdim=True), batch[0], batch[1]


class Encoder(nn.Module):
    def __init__(self, n_components):
        super(Encoder, self).__init__()
        config_bert = BertConfig(vocab_size = 10000,num_hidden_layers=2)
        self.encoder = BertForMaskedLM(config_bert)
       # self.encoder.bert.embeddings.requires_grad_(False)
        self.n_components = n_components

  
    def forward(self, x):
        input_ids = x['input_ids'].to(config.device)
        attention_mask = x["attention_mask"].to(config.device)
        labels = x["labels"].to(config.device)
        N = x['token_len'].to(config.device)
        outputs = self.encoder(input_ids = input_ids, attention_mask= attention_mask, labels = labels, output_hidden_states=True)
        loss = outputs[0]  #mlm
        hiden_state = outputs.hidden_states[-1]
        avg_pool = []
        for i, hiden_state in enumerate(hiden_state):
          text_rep = hiden_state[0:N[i]]
          text_rep = (torch.sum(text_rep, 0).reshape(1,-1)) / N[i]
          avg_pool.append(text_rep)
        
        avg_pool = torch.cat(avg_pool)
        
        
        return loss, avg_pool

    def predict(self, x):
        input_ids = x['input_ids'].to(config.device)
        attention_mask = x["attention_mask"].to(config.device)
        N = x['token_len'].to(config.device)
        outputs = self.encoder(input_ids = input_ids, attention_mask= attention_mask, output_hidden_states=True)
        hiden_state = outputs.hidden_states[-1]
        avg_pool = []
        for i, hiden_state in enumerate(hiden_state):
          text_rep = hiden_state[0:N[i]]
          text_rep = (torch.sum(text_rep, 0).reshape(1,-1)) / N[i]
          avg_pool.append(text_rep)
        
        avg_pool = torch.cat(avg_pool)

        return None,avg_pool
        
        
def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch**2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()
