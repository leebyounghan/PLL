#!/uㅇsr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.covariance import ledoit_wolf, ShrunkCovariance, OAS, EmpiricalCovariance
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import roc_curve, auc
from sklearn.svm import OneClassSVM
import config
import os
import pdb


def OC_SVM(ftrain, ftest, food):
    cls = OneClassSVM(gamma='auto')
    train_x = ftrain
    cls.fit(train_x)
    print("predicting test_in")
    test_in_pred = cls.score_samples(ftest)
    test_out_pred = cls.score_samples(food)

    return test_in_pred, test_out_pred


#MD one
def get_scores_one_cluster(ftrain, ftest, food, b, shrunkcov=True):
    if shrunkcov == "lw":
        print("Using ledoit-wolf covariance estimator.")
        cov = lambda x: ledoit_wolf(x)[0]
    elif shrunkcov == "emp":
        cov = lambda x: EmpiricalCovariance().fit(x).covariance_
    elif shrunkcov == "sh":
        cov = lambda x: ShrunkCovariance(shrinkage=0.1).fit(x).covariance_
    elif shrunkcov == "osa":
        cov = lambda x: OAS().fit(x).covariance_
    else:
        cov = lambda x: np.cov(x.T, bias=True)

    dtrain = np.sum(
        (ftrain - np.mean(ftrain, axis=0, keepdims=True)) *
        (np.linalg.pinv(cov(ftrain)).dot(
            (ftrain - np.mean(ftrain, axis=0, keepdims=True)).T)).T,
        axis=-1,
    )
    # ToDO: Simplify these equations
    dtest = np.sum(
        (ftest - np.mean(ftrain, axis=0, keepdims=True)) *
        (np.linalg.pinv(cov(ftrain)).dot(
            (ftest - np.mean(ftrain, axis=0, keepdims=True)).T)).T,
        axis=-1,
    )

    dood = np.sum(
        (food - np.mean(ftrain, axis=0, keepdims=True)) *
        (np.linalg.pinv(cov(ftrain)).dot(
            (food - np.mean(ftrain, axis=0, keepdims=True)).T)).T,
        axis=-1,
    )
    #  if shrunkcov == "lw":
    #  np.save(os.path.join(config.sub_log_path, "train_emb.txt" + b), ftrain)
    #  np.save(os.path.join(config.sub_log_path, "in_emb.txt" + b), ftest)
    #  np.save(os.path.join(config.sub_log_path, "ood_emb.txt" + b), food)
    #
    return dtest, dood, dtrain


#MD multi
def get_scores_multi_cluster(ftrain, ftest, food, ypred, b, shrunkcov=True):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]
    if shrunkcov == "lw":
        print("Using ledoit-wolf covariance estimator.")
        cov = lambda x: ledoit_wolf(x)[0]
    elif shrunkcov == "emp":
        cov = lambda x: EmpiricalCovariance().fit(x).covariance_
    elif shrunkcov == "sh":
        cov = lambda x: ShrunkCovariance(shrinkage=0.1).fit(x).covariance_
    elif shrunkcov == "osa":
        cov = lambda x: OAS().fit(x).covariance_
    else:
        cov = lambda x: np.cov(x.T, bias=True)

    dtrain = np.sum(
        (ftrain - np.mean(ftrain, axis=0, keepdims=True)) *
        (np.linalg.pinv(cov(ftrain)).dot(
            (ftrain - np.mean(ftrain, axis=0, keepdims=True)).T)).T,
        axis=-1,
    )

    din = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True)) *
            (np.linalg.pinv(cov(ftrain)).dot(
                (ftest - np.mean(x, axis=0, keepdims=True)).T)).T,
            axis=-1,
        ) for x in xc
    ]
    dood = [
        np.sum(
            (food - np.mean(x, axis=0, keepdims=True)) *
            (np.linalg.pinv(cov(ftrain)).dot(
                (food - np.mean(x, axis=0, keepdims=True)).T)).T,
            axis=-1,
        ) for x in xc
    ]

    din = np.min(din, axis=0)
    dood = np.min(dood, axis=0)
    cov_mat = np.array([cov(x) for x in xc])
    if shrunkcov == "lw":
        np.save(os.path.join(config.sub_log_path, "in_emb.txt" + b), ftest)
        np.save(os.path.join(config.sub_log_path, "ood_emb.txt" + b), food)
        np.save(os.path.join(config.sub_log_path, "y_label.txt" + b), ypred)
        np.save(os.path.join(config.sub_log_path, "train_emb.txt" + b), ftrain)
    return din, dood, dtrain


#CD
def get_scores_CD(ftrain, ftest, food):

    ftrain_avg = np.mean(ftrain, axis=0, keepdims=True)
    dtrain = [
        1 - cosine_similarity(i.reshape(1, -1), ftrain_avg) for i in ftrain
    ]
    din = [1 - cosine_similarity(i.reshape(1, -1), ftrain_avg) for i in ftest]
    dood = [1 - cosine_similarity(i.reshape(1, -1), ftrain_avg) for i in food]

    return np.array(din), np.array(dood), np.array(dtrain)


#UD
def get_scores_UD(ftrain, ftest, food):

    ftrain_avg = np.mean(ftrain, axis=0, keepdims=True)
    dtrain = [euclidean_distances(i.reshape(1, -1), ftrain_avg) for i in ftest]
    din = [euclidean_distances(i.reshape(1, -1), ftrain_avg) for i in ftest]
    dood = [euclidean_distances(i.reshape(1, -1), ftrain_avg) for i in food]

    return np.array(din), np.array(dood), np.array(dtrain)


def auroc_score(n, score):
    fpr, tpr, thresholds = roc_curve(n, score, pos_label=0)
    auroc = auc(fpr, tpr)
    return auroc


def get_mahalnobis_score(ftrain, ftest, food, labels):
    n = [0 for i in range(len(ftest))]
    N = [1 for i in range(len(food))]
    n.extend(N)

    din, dood, dtrain = get_scores_one_cluster(ftrain,
                                               ftest,
                                               food,
                                               "a",
                                               shrunkcov="lw")
    score_one = np.append(din, dood)
    config.one_lw = auroc_score(n, -score_one)

    din, dood, _ = get_scores_multi_cluster(ftrain,
                                            ftest,
                                            food,
                                            labels,
                                            "a",
                                            shrunkcov="lw")
    score_multi = np.append(din, dood)
    config.multi_lw = auroc_score(n, -score_multi)
    din, dood = OC_SVM(ftrain, ftest, food)
    score_svm = np.append(din, dood)
    config.oc_svm = auroc_score(n, score_svm)

    print(f"one class(auroc): {config.one_class}")
    print(f"one multi(auroc): {config.multi_class}")
    print(f"one svm(auroc): {config.oc_svm}")
