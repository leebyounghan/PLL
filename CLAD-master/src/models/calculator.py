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


def metric_calculate(net,
               train_x,
               test_in,
               test_out,
               test_ood,
               cluster_assigned):

    print("in-distribution data")
    train_x_loader = DataLoader(train_x, batch_size=1, shuffle=False)
    test_in_loader = DataLoader(test_in, batch_size=1, shuffle=False)
    test_out_loader = DataLoader(test_out, batch_size=1, shuffle=False)
    test_ood_loader = DataLoader(test_ood, batch_size=1, shuffle=False)

    maha = {}
    msp = {}
    energy = {}

    train_x_emb = []
    test_in_emb = []
    test_out_emb = []
    test_ood_emb = []

    energy_in = []
    energy_out = []
    energy_ood = []

    msp_in = []
    msp_out = [] 
    msp_ood = []

    print("preparing train data feature") 
    for j, data in enumerate(train_x_loader):
        #base
        if config.classifier_type in bert_classifier:
            net.eval()
            outputs, inputs, hiden_state = net(
                data['input_ids'].to(config.device),
                data['attention_mask'].to(config.device), 1, None)
            train_x_emb.append(hiden_state.detach().cpu())


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
    print("out-of-distribution(in-domain) data")


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


    print("out-of-distribution(out-domain) data")
    for j, data in enumerate(test_ood_loader):

        #base
        if config.classifier_type in bert_classifier:
            net.eval()
            outputs, inputs, hiden_state = net(
                data['input_ids'].to(config.device),
                data['attention_mask'].to(config.device), 1, None)
            test_ood_emb.append(hiden_state.detach().cpu())

    
        nnOutputs = outputs
        energy_ood.append(torch.logsumexp(nnOutputs, dim=-1).data.cpu())
        
        nnOutputs = F.softmax(nnOutputs, dim=-1).max(-1)[0].data.cpu()
        msp_ood.append(nnOutputs)

        if j % 100 == 99:
            print("{:4}/{:4} data processed, {:.1f} seconds used.".format(
                j + 1,
                #  test_num,
                len(test_ood_loader),
                time.time() - t0))
            t0 = time.time()


    maha["train"] = torch.cat(train_x_emb).numpy()
    maha["test_in"] = torch.cat(test_in_emb).numpy()
    maha["test_out"] = torch.cat(test_out_emb).numpy()
    maha['test_ood'] = torch.cat(test_ood_emb).numpy()


    maha = get_scores_multi_cluster(maha, cluster_assigned, shrunkcov = True)


    msp["test_in"] = torch.cat(msp_in).numpy()
    msp["test_out"] = torch.cat(msp_out).numpy()
    msp['test_ood'] = torch.cat(msp_ood).numpy()

    energy["test_in"] = torch.cat(energy_in).numpy()
    energy["test_out"] = torch.cat(energy_out).numpy()
    energy['test_ood'] = torch.cat(energy_ood).numpy()
    
    result = []

    result.append(metric(energy["test_in"], energy["test_out"], ["energy"]))
    result.append(metric(energy['test_in'], energy['test_ood'], ['energy_ood']))
    
    result.append(metric(maha["test_in"], maha["test_out"], ["maha"]))
    result.append(metric(maha['test_in'], maha['test_ood'], ['maha_ood']))
    
    result.append(metric(msp["test_in"], msp["test_out"], ["msp"]))
    result.append(metric(msp['test_in'], msp['test_ood'], ['msp_ood']))

    log_df(result)
    return result



def get_scores_multi_cluster(maha, ypred, shrunkcov='lw'):
    ftrain = maha['train']
    ftest, fout, food = maha['test_in'], maha['test_out'], maha['test_ood']
    
    np.save(os.path.join(config.sub_log_path, 'train_embed.txt'), ftrain)
    np.save(os.path.join(config.sub_log_path, 'in_embed.txt'), ftrain)
    np.save(os.path.join(config.sub_log_path, 'out_embed.txt'), ftest)
    np.save(os.path.join(config.sub_log_path, 'ood_embed.txt'), food)


    xc = [ftrain[ypred == i] for i in np.unique(ypred)]
    if shrunkcov =="lw":
        print("Using ledoit-wolf covariance estimator.")
        cov = np.linalg.pinv(ledoit_wolf(ftrain)[0])
    else:
        print("nor")
        cov = np.linalg.pinv(np.cov(ftrain.T, bias=True))

    #train maha-dist
    dtrain = [
        np.sum(
            (ftrain - np.mean(x, axis=0, keepdims=True))
            * (
                cov.dot(
                    (ftrain - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]       
    
    #test-in maha-dist
    din = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True))
            * (
                cov.dot(
                    (ftest - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    #test ood(in-domain) dist
    dout = [
        np.sum(
            (fout - np.mean(x, axis=0, keepdims=True))
            * (
                cov.dot(
                    (fout - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    # test ood(out-domain) dist

    dood = [
        np.sum(
            (food - np.mean(x, axis=0, keepdims=True))
            * (
                cov.dot(
                    (food - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]


    din = np.min(din, axis=0)
    dout = np.min(dout, axis=0)
    dood = np.min(dood, axis=0)
    dtrain = np.min(dtrain, axis=0)

    return {'test_in': din, 'test_out':dout, 'test_ood': dood}



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




def log_df(result):
    for i in result:
        key = list(i.keys())[0]
        
        result = pd.read_csv(f"./{config.dataset_name}/result.csv")
        config_condition = {"Date":config.today,
                           "time":config.current_time,
                           'score_func' : key,
                           'Dataset': config.dataset_name,
                           "cluster_type":config.cluster_type,
                           "cluster_epochs":config.cluster_model_train_epochs,
                           "cluster_lr":config.cluster_model_train_lr,
                           "classifier_epochs":config.classifier_epochs,
                           "cluster_num":config.cluster_num,
                           "Normal_class":config.normal_class_index_list,
                           "AUROC" :  i[key]['AUROC'],
                           "AUIN" : i[key]['AUIN'],
                           "AUOUT" : i[key]['AUOUT'],
                           "DTACC" : i[key]['DTACC'],
                           "cluster_acc":config.cluster_acc,
                           "pooling": config.pooling}

        df_new = pd.DataFrame.from_dict(config_condition)
        result.append(df_new,ignore_index=True).to_csv(f"./{config.dataset_name}/result.csv", index=False)

#  def get_scores_multi_cluster(ftrain, ftest, food, ypred, b, shrunkcov=True):
    #  xc = [ftrain[ypred == i] for i in np.unique(ypred)]
    #  if shrunkcov == "lw":
        #  print("Using ledoit-wolf covariance estimator.")
        #  cov = ledoit_wolf(x)[0]
    #  else:
        #  cov = lambda x: np.cov(x.T, bias=True)
#
#
#
    #  din = [
        #  np.sum(
            #  (ftest - np.mean(x, axis=0, keepdims=True)) *
            #  (np.linalg.pinv(cov).dot(
                #  (ftest - np.mean(x, axis=0, keepdims=True)).T)).T,
            #  axis=-1,
        #  ) for x in xc
    #  ]
    #  dood = [
        #  np.sum(
            #  (food - np.mean(x, axis=0, keepdims=True)) *
            #  (np.linalg.pinv(cov).dot(
                #  (food - np.mean(x, axis=0, keepdims=True)).T)).T,
            #  axis=-1,
        #  ) for x in xc
    #  ]
#
    #  din = np.min(din, axis=0)
    #  dood = np.min(dood, axis=0)
#
    #  return din, dood
#
