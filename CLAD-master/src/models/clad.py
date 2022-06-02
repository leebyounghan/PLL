# -*- coding: utf-8 -*-
from models.clustering_model import Clustering_Module, binary_cluster_accuracy
from models.DEC_bert_test import Clustering_Module_bert, testDataset, set_text_data
#from models.DEC_nopretrain import Clustering_Module_bert_scr, set_text_predict_cls
#  from models.linear_clustering import LinearClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from .confidence_trainers import FC3_classifier, BERT, BERT_mlm, BERT_cls
from .utils import plot_distribution, cluster_accuracy
#from .utils import draw_plot
from config import implemented_cluster_models, implemented_classifier_models
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from itertools import combinations
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
import torch
import torchvision
from torchvision import transforms
from data_util.utils import divide_data_label
from data_util.embeddings import bert_test_data, testDataset, NewsDataset

#from models.odin import apply_odin
from models.odin_maha import apply_odin
#  from models.new import apply_odin
from models.mahalanobis import get_scores_one_cluster, get_scores_multi_cluster, get_mahalnobis_score
from models.metric import calculate_metric
import models.SCCL as SCCL
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
#from models.utils import plot_distribution
import pandas as pd
import os
import numpy as np
import config
import pdb
from collections import Counter
from sklearn.metrics import pairwise_distances
from sklearn.utils import resample

# logger
log = config.logger
log_path = config.log_path
if os.path.exists(log_path) == False:
    os.makedirs(log_path)
sub_log_path = config.sub_log_path
if os.path.exists(sub_log_path) == False:
    os.makedirs(sub_log_path)




class CLAD(object):
    """
    self.train_pred_label = predicted label from clustering model
    self.val_pred_label =
    self.test_pred_label =
    self.cluster_model =
    """
    def __init__(self, dataset_name, dataset, cluster_num, cluster_type,
                 classifier_type):
        """TODO: to be defined. """
        self.dataset_name = dataset_name
        self.train_x = dataset["train_x"]
        self.train_y = dataset["train_y"]
        self.test_in = dataset["test_in"]
        self.test_out = dataset["test_out"]
        self.train_text = dataset['train_text']
        self.test_in_emb = dataset['test_in_emb']
        self.test_out_emb = dataset['test_out_emb']

        # cluster variables
        self.clusters = self.train_y
        self.cluster_num = cluster_num
        self.cluster_type = cluster_type
        self.train_clusters = []
        self.cluster_model = None
        self.center = None
        # classifier variables
        self.classifier_type = classifier_type
        self.cluster_model = None
        assert cluster_type in implemented_cluster_models
        assert classifier_type in implemented_classifier_models

    def cluster(self):
        """
        clustering module of the model
        """

        log = config.logger
        if self.cluster_type == 'linear_word':
            print("linear_word")
            cluster_model = Clustering_Module(
                dataset_name=self.dataset_name,
                train_x=self.train_x,
                train_y=self.train_y,
                test_in=self.test_in,
                test_out=self.test_out,
                batch_size=config.cluster_model_batch_size,
                cluster_type=self.cluster_type,
                n_components=self.cluster_num,
                n_hidden_features=config.n_hidden_features)

            cluster_model.pretrain(epochs=config.cluster_model_pretrain_epochs)
            cluster_model.train(epochs=config.cluster_model_train_epochs)
            self.clusters, emb_normal, _ = cluster_model.predict()
            #self.clusters_out, emb_ab,_ = cluster_model.predict_out()

        elif self.cluster_type == "SCCL":
            self.clusters, self.bertmlm = SCCL.run(self.train_text,
                                                   self.train_y)
            labels = self.clusters.cpu().numpy()
            config.cluster_acc = cluster_accuracy(self.train_y, labels)[1]
            print(config.cluster_acc)
        else:
            #if self.cluster_type =='DEC_bert':
            print("DEC_bert")
            cluster_model = Clustering_Module_bert(
                self.train_text,
                self.test_in,
                self.test_out,
                self.cluster_num,
                batch_size=config.cluster_model_batch_size)

            cluster_model.train(epochs=config.cluster_model_train_epochs)
            self.clusters, emb_normal, km_pred = cluster_model.predict()
            clusters_in, emb_in, clusters_out, emb_out = cluster_model.predict_out(
            )

            if config.gt == True:
                self.clusters = self.train_y

            labels = self.clusters.numpy()
            print(Counter(labels))
            #config.normal_distance = distance
            self.bertmlm = cluster_model.cm.encoder.encoder
            #  din, dout, dtrain = get_scores_one_cluster(emb_normal.numpy(),
            #  emb_in.numpy(),
            #  emb_out.numpy(), "b",
            #  "lw")
            #  din_m, dout_m, dtrain_m = get_scores_multi_cluster(
            #  emb_normal.numpy(), emb_in.numpy(), emb_out.numpy(), labels,
            #  "b", "sh")
            #  n = [0 for i in range(len(emb_in))]
            #  N = [1 for i in range(len(emb_out))]
            #  n.extend(N)
            #  score_one = np.append(din, dout)
            #  fpr, tpr, thresholds = roc_curve(n, -score_one, pos_label=0)
            #  auroc_one = auc(fpr, tpr)
            #  score_multi = np.append(din_m, dout_m)
            #  fpr, tpr, thresholds = roc_curve(n, -score_multi, pos_label=0)
            #  auroc_multi = auc(fpr, tpr)
            #  label_true = self.train_y.numpy()
            #  config.before_PLL_s = auroc_one
            #  config.before_PLL_m = auroc_multi
            if config.cluster_num > 1:
                #  NMI = normalized_mutual_info_score(label_true, labels)
                #  ari = adjusted_rand_score(label_true, labels)
                config.cluster_acc = cluster_accuracy(self.train_y, labels)[1]
            #  precomputed = pairwise_distances(emb_normal.numpy(), metric='mahalanobis')
            #  config.silhouette_score_b = silhouette_score(
            #  emb_normal, labels)
            #  precomputed = pairwise_distances(emb_normal.numpy(), metric='mahalanobis')
            #  config.NMI = NMI
            #  config.ARI = ari
            #  config.db_score = davies_bouldin_score(emb_normal, labels)
            #  config.ch_score = calinski_harabasz_score(emb_normal, labels)
#
        print("end cluster in clad")
        if (config.classifier_type in ['dec']):
            tokenizer = AutoTokenizer.from_pretrained(config.LM)
            test_in_encodings = tokenizer(self.test_in,
                                          max_length=config.max_length,
                                          truncation=True,
                                          padding=True)
            test_out_encodings = tokenizer(self.test_out,
                                           max_length=config.max_length,
                                           truncation=True,
                                           padding=True)
            in_data_len = len(self.test_in)
            out_data_len = len(self.test_out)
            self.test_in = testDataset(test_in_encodings,
                                       in_data_len)  #make test data token
            self.test_out = testDataset(test_out_encodings,
                                        out_data_len)  #make test data token
            apply_odin(cluster_model.cm, self.test_in, self.test_out, None,
                       None, None)
            calculate_metric()

    def classify_nn(self, dataset_name):
        #  TODO: implement save / load of classifier model
        log = config.logger
        classifier_type = self.classifier_type
        assert classifier_type in implemented_classifier_models

        if (classifier_type == 'linear'):
            print('linear')
            classifier = Linear_classifier(
                self.train_x,
                self.clusters,
                n_epochs=config.classifier_epochs,
                lr=config.classifier_lr)

        elif (classifier_type == 'fc3'):
            print('fc3')
            #                 if use_DECencoder:
            #                     self.train_x = encoder(self.train_x)
            classifier = FC3_classifier(self.train_x,
                                        self.clusters,
                                        n_epochs=config.classifier_epochs,
                                        lr=config.text_classifier_lr)

        elif (classifier_type == 'bert'):
            print('bert')
            tokenizer = AutoTokenizer.from_pretrained(config.LM)
            train_encodings = tokenizer(self.train_text,
                                        max_length=config.max_length,
                                        truncation=True,
                                        padding=True)  #max_length = 512
            train_dataset = NewsDataset(train_encodings, self.clusters)
            classifier = BERT(train_dataset,
                              n_epochs=config.classifier_epochs,
                              lr=config.text_classifier_lr)

        elif (classifier_type == 'DEC_bert'):
            print('DEC_bert')
            tokenizer = AutoTokenizer.from_pretrained(config.LM)

            train_encodings = tokenizer(
                self.train_text,
                truncation=True,
                padding=True,
                max_length=config.max_length)  #max_length = 512
            train_dataset = NewsDataset(train_encodings, self.clusters)
            classifier = BERT_mlm(train_dataset,
                                  n_epochs=config.classifier_epochs,
                                  lr=config.text_classifier_lr,
                                  models=self.bertmlm,
                                  batch=config.cluster_model_batch_size)

            #  if(classifier_type in  ["bert","DEC_bert","bert_scr"]):
            #  train_pred = classifier.predict(train_dataset)
            #  else:
            #  train_pred = classifier.predict(self.train_x.cuda(config.device))


#
#  train_accuracy = accuracy_score(train_pred, self.clusters)
#
#
#        confidence trainer accuracy
#  print("Calculating NN Classifier training accuracy...")
#  print("NN Classifier training accuracy : {}".format(train_accuracy))
#  log.info("NN Classifier training accuracy = {}".format(train_accuracy))
#

        if config.classifier_type in ['bert', 'DEC_bert', "bert_scr"]:
            in_data_len = len(self.test_in)
            out_data_len = len(self.test_out)
            self.train_x = bert_test_data(self.train_text)
            self.test_in = bert_test_data(self.test_in)  #make test data token
            self.test_out = bert_test_data(
                self.test_out)  #make test data token

        print("Scailing the confidence outputs")
        #if fc self.test.emd X
        #apply_odin(classifier, self.test_in, self.test_out,self.center,self.test_in_emb,self.test_out_emb)

        train_x_emb, test_in_emb, test_out_emb = apply_odin(
            classifier, self.train_x, self.test_in, self.test_out, self.center,
            self.test_in_emb, self.test_out_emb)

        if config.cluster_num > 1:
            config.silhouette_score_a = silhouette_score(
                train_x_emb,
                self.clusters.cpu().numpy())
        #score_one
        # emb_s = torch.cat((self.test_in_emb, self.test_out_emb))
        # plot_distribution(test_in_emb,test_out_emb,score_one,score_multi,emb_s)

        get_mahalnobis_score(train_x_emb, test_in_emb, test_out_emb,
                             self.clusters.cpu().numpy())

        print("Calculating Metrics")
        calculate_metric()
