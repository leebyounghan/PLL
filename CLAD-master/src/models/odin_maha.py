#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from transformers import DataCollatorForLanguageModeling, BertTokenizer
from models.utils import plot_distribution
import numpy as np
import time
import os

import config

from data_util.embeddings import bert_test_data, testDataset, NewsDataset

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

    test_num = max(len(test_in), len(test_out))
    score_ = []
    criterion = nn.CrossEntropyLoss()
    t0 = time.time()
    if os.path.exists(config.sf_scores_path) == False:
        os.makedirs(config.sf_scores_path)
    f1 = open(config.base_in_path, "w")
    f2 = open(config.base_out_path, "w")
    g1 = open(config.odin_in_path, "w")
    g2 = open(config.odin_out_path, "w")

    temper = config.temperature
    noise_magnitude = config.perturbation
    cls_emb = []
    lab = []
    train_x_emb = []
    test_in_emb = []
    test_out_emb = []
    for j, data in enumerate(train_x_loader):
        #base
        if config.classifier_type in bert_classifier:
            net.eval()
            #outputs, inputs = net(data['input_ids'].to(config.device),data['attention_mask'].to(config.device),1, None)
            outputs, inputs, hiden_state = net(
                data['input_ids'].to(config.device),
                data['attention_mask'].to(config.device), 1, None)
            train_x_emb.append(hiden_state.detach().cpu())

    for j, data in enumerate(test_in_loader):
        #base
        if config.classifier_type in bert_classifier:
            net.eval()
            #outputs, inputs = net(data['input_ids'].to(config.device),data['attention_mask'].to(config.device),1, None)
            outputs, inputs, hiden_state = net(
                data['input_ids'].to(config.device),
                data['attention_mask'].to(config.device), 1, None)
            test_in_emb.append(hiden_state.detach().cpu())

        else:
            inputs = Variable(data.float().cuda(config.device),
                              requires_grad=True)
            net.eval()
            outputs = net(inputs)
        #outputs = outputs.reshape(1,-1)

        # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()

        if config.classifier_type in ['dec']:
            pass

        else:
            nnOutputs = nnOutputs.reshape(
                1, -1)  #*distence.reshape(1, -1) #가까운 클러스터에 더 많은 가중치 부여
            nnOutputs = nnOutputs - np.max(nnOutputs)
            nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
            #s = np.multiply(nnOutputs.reshape(-1),distence.reshape(-1))

        f1.write("{}, {}, {}\n".format(temper, noise_magnitude,
                                       np.max(nnOutputs)))
        score_.append(np.max(nnOutputs))

        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t input
        maxIndexTemp = np.argmax(nnOutputs)
        lab.append(maxIndexTemp)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(config.device))
        loss = criterion(outputs, labels)
        loss.backward()
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient = gradient * 50 / (79.0 / 255.0)

        #odin
        if config.classifier_type in bert_classifier:
            outputs, inputs_, _ = net(data['input_ids'].to(config.device),
                                      data['attention_mask'].to(config.device),
                                      1, gradient)
            outputs = outputs / temper

#        elif config.classifier_type == 'DEC_bert':
#            outputs, inputs = net(data)

        else:
            tempInputs = torch.add(inputs.data, -noise_magnitude, gradient)
            outputs = net(Variable(tempInputs))
            outputs = outputs / temper

        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()

        if config.classifier_type in ['dec']:
            outputs = outputs * temper

        else:
            nnOutputs = nnOutputs.reshape(-1)  #*distence.reshape(-1)
            nnOutputs = nnOutputs - np.max(nnOutputs)
            nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
            #s = np.multiply(nnOutputs.reshape(-1),distence.reshape(-1))

        g1.write("{}, {}, {}\n".format(temper, noise_magnitude,
                                       np.max(nnOutputs)))

        if j % 100 == 99:
            print("{:4}/{:4} data processed, {:.1f} seconds used.".format(
                j + 1,
                #  test_num,
                len(test_in_loader),
                time.time() - t0))
            t0 = time.time()

        if j > test_num:
            break

        torch.cuda.empty_cache()

    # out distribution test
    print("out-of-distribution data")

    normal_cls = len(cls_emb)

    for j, data in enumerate(test_out_loader):

        #base
        if config.classifier_type in bert_classifier:
            net.eval()
            outputs, inputs, hiden_state = net(
                data['input_ids'].to(config.device),
                data['attention_mask'].to(config.device), 1, None)
            test_out_emb.append(hiden_state.detach().cpu())
            #outputs, inputs = net(data['input_ids'].to(config.device))

        else:
            inputs = Variable(data.cuda(config.device),
                              requires_grad=True)  #오딘은 무조건 숫자여야함
            net.eval()
            outputs = net(inputs)

        #outputs = outputs.reshape(1,-1)
        # Calculating the confidence of the output, no pertyrbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        if config.classifier_type in ['dec']:
            pass

        else:
            nnOutputs = nnOutputs.reshape(1, -1)  #*distence.reshape(1,-1)
            nnOutputs = nnOutputs - np.max(nnOutputs)
            nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
            #s = np.multiply(nnOutputs.reshape(-1),distence.reshape(-1))

        f2.write("{}, {}, {}\n".format(temper, noise_magnitude,
                                       np.max(nnOutputs)))
        score_.append(np.max(nnOutputs))

        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        lab.append(maxIndexTemp)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(config.device))
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient = gradient * 50 / (79.0 / 255.0)

        if config.classifier_type in bert_classifier:
            outputs, inputs_, _ = net(data['input_ids'].to(config.device),
                                      data['attention_mask'].to(config.device),
                                      1, gradient)

            outputs = outputs / temper

        else:
            tempInputs = torch.add(inputs.data, -noise_magnitude, gradient)
            outputs = net(Variable(tempInputs))
            outputs = outputs / temper

        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()

        if config.classifier_type in ['dec']:  #dec는 확률로 나옴
            outputs = outputs * temper

        else:  #나머지는 score로 나옴
            nnOutputs = nnOutputs.reshape(-1)  #*distence.reshape(-1)
            nnOutputs = nnOutputs - np.max(nnOutputs)
            nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
            #s = np.multiply(nnOutputs.reshape(-1),distence.reshape(-1))

        g2.write("{}, {}, {}\n".format(temper, noise_magnitude,
                                       np.max(nnOutputs)))

        if j % 100 == 99:
            print("{:4}/{:4} data processed, {:.1f} seconds used.".format(
                j + 1,
                #  test_num,
                len(test_out_loader),
                time.time() - t0))
            t0 = time.time()

        if j > test_num:
            break

        torch.cuda.empty_cache()


#     emb=torch.cat((test_in,test_out)).numpy()
#     embed = pd.DataFrame(emb,columns=['emb_x',"emb_y"])
#     col = ["red","green","blue","yellow","pink","orange","purple","brown","cyan","magenta",
#        "lightblue","aqua",'tan','indigo','lime','tomato','maroon','gold','chartreuse','khaki'] #need different 20 coloers
#     normal_ = ["normal" for i in range(test_in.shape[0])]
#     abnormal_ = ["abnormal" for i in range(test_out.shape[0])]
#     normal_.extend(abnormal_)
#     pred_cls = [col[i] for i in lab]
#     embed["pred_cls"]=pred_cls
#     embed['score'] = score_
#     embed['label_gt'] = normal_
#     embed.to_csv(config.sub_log_path+"/sbert_emb.csv")
#     return embed

    train_x_emb = torch.cat(train_x_emb)
    test_in_emb = torch.cat(test_in_emb)
    test_out_emb = torch.cat(test_out_emb)
    return train_x_emb.numpy(), test_in_emb.numpy(), test_out_emb.numpy()
