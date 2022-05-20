#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.covariance import ledoit_wolf, ShrunkCovariance, OAS, EmpiricalCovariance
import numpy as np
import time
import os

import config
import pdb

bert_classifier = ["bert", "DEC_bert", "bert_scr"]


def apply_odin(net,
               train_x,
               test_in,
               test_out,
               center,
               test_in_emb=None,
               test_out_emb=None):

    print("in-distribution data")
    train_x_loader = DataLoader(train_x, batch_size=1, shuffle=False)
    test_in_loader = DataLoader(test_in, batch_size=1, shuffle=False)
    test_out_loader = DataLoader(test_out, batch_size=1, shuffle=False)


    maha = {}
    msp = {}
    energy = {}

    train_x_emb = []
    test_in_emb = []
    test_out_emb = []

    energy_in = []
    energy_out = []

    msp_in = []
    msp_out = [] 

    # for train_noraml data point
    for j, data in enumerate(train_x_loader):
        #base
    
        if config.classifier_type in bert_classifier:
            net.eval()
            outputs, inputs, hiden_state = net(
                data['input_ids'].to(config.device),
                data['attention_mask'].to(config.device), 1, None)
            train_x_emb.append(hiden_state.detach().cpu())

    #for in_dist

    for j, data in enumerate(test_in_loader):

        t0 = time.time()
        #base
        if config.classifier_type in bert_classifier:
            net.eval()
            outputs, inputs, hiden_state = net(
                data['input_ids'].to(config.device),
                data['attention_mask'].to(config.device), 1, None)
            test_in_emb.append(hiden_state.detach().cpu())


        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs
        energy_in.append(torch.logsumexp(nnOutputs, dim=-1).data.cpu())


        nnOutputs = F.softmax(nnOutputs, dim=-1).max(-1)[0].data.cpu()
        msp_in.append(nnOutputs)



        if j % 100 == 99:
            print("{:4}/{:4} data processed, {:.1f} seconds used.".format(
                j + 1,
                #  test_num,
                len(test_in_loader),
                time.time() - t0))
            t0 = time.time()


        torch.cuda.empty_cache()

    # out distribution test
    print("out-of-distribution data")


    for j, data in enumerate(test_out_loader):

        #base
        if config.classifier_type in bert_classifier:
            net.eval()
            outputs, inputs, hiden_state = net(
                data['input_ids'].to(config.device),
                data['attention_mask'].to(config.device), 1, None)
            test_out_emb.append(hiden_state.detach().cpu())

    
        nnOutputs = outputs
        energy_out.append(torch.logsumexp(nnOutputs, dim=-1).data.cpu())
        
        nnOutputs = F.softmax(nnOutputs, dim=-1).max(-1)[0].data.cpu()
        msp_out.append(nnOutputs)

        if j % 100 == 99:
            print("{:4}/{:4} data processed, {:.1f} seconds used.".format(
                j + 1,
                #  test_num,
                len(test_out_loader),
                time.time() - t0))
            t0 = time.time()




    maha["train"] = torch.cat(train_x_emb).numpy()
    maha["test_in"] = torch.cat(test_in_emb).numpy()
    maha["test_out"] = torch.cat(test_out_emb).numpy()

    pdb.set_trace()
    msp["test_in"] = torch.cat(msp_in).numpy()
    msp["test_out"] = torch.cat(msp_out).numpy()
    energy["test_in"] = torch.cat(energy_in).numpy()
    energy["test_out"] = torch.cat(energy_out).numpy()
    metric(energy["test_in"],energy["test_out"],"energy")
    return maha, msp, energy 



def get_scores_multi_cluster(ftrain, ftest, food, ypred, b, shrunkcov=True):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]
    if shrunkcov == "lw":
        print("Using ledoit-wolf covariance estimator.")
        cov = ledoit_wolf(x)[0]
    else:
        cov = lambda x: np.cov(x.T, bias=True)



    din = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True)) *
            (np.linalg.pinv(cov).dot(
                (ftest - np.mean(x, axis=0, keepdims=True)).T)).T,
            axis=-1,
        ) for x in xc
    ]
    dood = [
        np.sum(
            (food - np.mean(x, axis=0, keepdims=True)) *
            (np.linalg.pinv(cov).dot(
                (food - np.mean(x, axis=0, keepdims=True)).T)).T,
            axis=-1,
        ) for x in xc
    ]

    din = np.min(din, axis=0)
    dood = np.min(dood, axis=0)

    #  if shrunkcov == "lw":
    #  np.save(os.path.join(config.sub_log_path, "ood_emb.txt" + b), food)
    #  np.save(os.path.join(config.sub_log_path, "y_label.txt" + b), ypred)
    #
    return din, dood



def get_curve(in_dist, out_dist,stypes=['maha']):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known = -in_dist
        novel = -out_dist
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[stype][l + 1:] = tp[stype][l]
                fp[stype][l + 1:] = np.arange(fp[stype][l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l + 1:] = np.arange(tp[stype][l] - 1, -1, -1)
                fp[stype][l + 1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l + 1] = tp[stype][l]
                    fp[stype][l + 1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l + 1] = tp[stype][l] - 1
                    fp[stype][l + 1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95

def metric(in_dist, out_dist, stypes=['maha'], verbose=False):
    tp, fp, tnr_at_tpr95 = get_curve(in_dist, out_dist, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')

    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()

        # TNR
        mtype = 'TNR'
        results[stype][mtype] = tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[stype][mtype]),
                  end='')

        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype] / tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype] / fp[stype][0], [0.]])
        results[stype][mtype] = -np.trapz(1. - fpr, tpr)
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[stype][mtype]),
                  end='')

        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = .5 * (tp[stype] / tp[stype][0] + 1. -
                                      fp[stype] / fp[stype][0]).max()
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[stype][mtype]),
                  end='')

        # AUIN
        mtype = 'AUIN'
        denom = tp[stype] + fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype] / denom, [0.]])
        results[stype][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[stype][mtype]),
                  end='')

        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0] - tp[stype] + fp[stype][0] - fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0] - fp[stype]) / denom, [.5]])
        results[stype][mtype] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100. * results[stype][mtype]),
                  end='')
            print('')
    return results



#3가지로 만들어서 출력하면 끄읕

# stype = ["maha", "msp", "erg"]
# score = [maha,msp, erg]
# for name, score in zip(stpye, score):
#   results = metric(in_dist=score[test_in] , out_dist=score[test_out], stypes = name)
#   results 처리 + 



    # result = pd.read_csv("result.csv")
    # config_condition = {"Date":[config.today],
    #                     "time":[config.current_time],
    #                     "cluster_type":[config.cluster_type],
    #                     "cluster_epochs":[config.cluster_model_train_epochs],
    #                     "cluster_lr":[config.cluster_model_train_lr],
    #                     "classifier_epochs":[config.classifier_epochs],
    #                     "cluster_num":[config.cluster_num],
    #                     "Normal_class":[config.normal_class_index_list],
    #                     "one_ class" : config.one_class,
    #                     "multi_ class" : config.multi_class,
    #                     "before_PLL_s": config.before_PLL_s,
    #                     "before_PLL_m": config.before_PLL_m,
    #                     "OC_SVM" : config.oc_svm,
    #                     "normal_distance": config.normal_distance,
    #                     "cosin distance": config.cd_distance,
    #                     "cluster_acc":config.cluster_acc,
    #                     "pooling": config.pooling,
    #                     "one_lw": config.one_lw,
    #                     "one_emp" : config.one_emp,
    #                     "one_sh" : config.one_sh,
    #                     "one_osa" : config.one_osa,
    #                     "multi_lw" : config.multi_lw,

    #                     }
    # df_new = pd.DataFrame(config_condition)
    # result.append(df_new,ignore_index=True).to_csv("result.csv", index=False)

