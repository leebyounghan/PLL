#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch.nn.utils as torch_utils

from models.ptsdae.sdae import StackedDenoisingAutoEncoder
import models.ptsdae.model as ae

from models.ptdec.dec import DEC
from models.ptdec.model import train, predict

import config

import pdb

class LinearClustering():
    def __init__(self, dataset_name, train_x, train_y, batch_size, n_components, n_hidden_features):
        self.cuda = config.device
        self.dataset_name = dataset_name
        self.input_dim = len(train_x[0])
        self.ds_train = CachedData(data_x = train_x,
                                   data_y = train_y,
                                   cuda = self.cuda)
        self.batch_size = batch_size
        self.n_components = n_components
        self.n_hidden_features = n_hidden_features

    def pretrain(self):
        self.sdae = StackedDenoisingAutoEncoder(
            [self.input_dim, 500, 500, 2000, self.n_components],
            final_activation=None)


        # gradient clipping
        max_grad_norm = 3.
        torch_utils.clip_grad_norm_(self.sdae.parameters(), max_grad_norm)

        if self.cuda:
            self.sdae.cuda()
        print('pretraining stacked denoising autoencoder')
        ae.pretrain(
            self.ds_train,
            self.sdae,
            cuda = self.cuda,
            epochs = config.cluster_model_pretrain_epochs,
            batch_size = config.cluster_model_batch_size,
            optimizer = lambda model: SGD(model.parameters(), lr=config.cluster_model_pretrain_lr,
                                          momentum=config.cluster_model_pretrain_momentum),
            scheduler = lambda x: StepLR(x, 100, gamma=0.1),
            corruption=0.2
        )


        print('training stacked denoising autoencoder')
        ae.train(
            self.ds_train,
            self.sdae,
            cuda = self.cuda,
            epochs=config.cluster_model_pretrain_epochs,
            batch_size=config.cluster_model_batch_size,
            optimizer=SGD(self.sdae.parameters(),lr = config.cluster_model_train_lr,
                                        momentum=config.cluster_model_train_momentum),
            #scheduler=StepLR(optimizer, 100, gamma=0.1),
            corruption=0.2
        )
        
    def train(self):
        print('training linear clustering model')
        self.linear_clustering = DEC(cluster_number=self.n_components,
                                     hidden_dimension=self.n_components,
                                     encoder=self.sdae.encoder)
        if self.cuda:
            self.linear_clustering.cuda()
        lc_optimizer = SGD(self.linear_clustering.parameters(),
                           lr = config.cluster_model_lr,
                           momentum=config.cluster_model_momentum)
        train(dataset=self.ds_train,
              model=self.linear_clustering,
              epochs=config.cluster_model_epochs,
              batch_size=256,
              optimizer=lc_optimizer,
              stopping_delta=0.000001,
              cuda=self.cuda)

    def predict(self):
        log = config.logger

        train_predicted, train_actual = predict(self.ds_train,
                                                self.linear_clustering,
                                                1024,
                                                silent=True,
                                                return_actual=True,
                                                cuda=self.cuda)
        train_predicted = train_predicted.cpu().numpy()
        train_actual = train_actual.cpu().numpy()

        train_predicted = np.array(train_predicted)

        return train_predicted


class CachedData(Dataset):
    def __init__(self, data_x, data_y, cuda, testing_mode=False):
        if not cuda:
            data_x.detach().cpu()
            data_y.detach().cpu()
        self.ds = TensorDataset(data_x, data_y)
        self.cuda = cuda
        self.testing_mode = testing_mode
        self._cache = dict()

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.ds[index])
            if self.cuda:
                self._cache[index][0] = self._cache[index][0].cuda(non_blocking=True)
                self._cache[index][1] = self._cache[index][1].cuda(non_blocking=True)
        return self._cache[index]

    def __len__(self) -> int:
        return 128 if self.testing_mode else len(self.ds)



