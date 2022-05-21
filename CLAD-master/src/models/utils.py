#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
#  from sklearn.manifold import TSNE
#from MulticoreTSNE import MulticoreTSNE as TSNE
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import config

import pdb
import matplotlib

def plot_distribution(indist, ood, score_one, score_multi,sbert):

    print("plotting image on " + config.plot_path + "...")
    if (os.path.exists(config.plot_path) == False):
        os.makedirs(config.plot_path)

        
    n = ["normal" for i in range(len(indist))]
    N = ["abnormal" for i in range(len(ood))]
    n.extend(N)
    data_x = np.vstack([indist,ood])
    embed = pd.DataFrame(mapper.fit_transform(data_x), columns=['x', 'y'])
    
    embed['label_gt'] = n
    embed['score_one'] = score_one
    embed['score_multi'] = score_multi
    scatter_plot(embed)


def scatter_plot(embed):
    if (os.path.exists(config.plot_path) == False):
        os.makedirs(config.path)

    plt.figure(figsize=(20, 15))
    fig = sns.scatterplot(data = embed, x='x',y="y", hue="label_gt", size ='score_one')
    plt.setp(fig.get_legend().get_texts(), fontsize='10')
    plt.setp(fig.get_legend().get_title(), fontsize='10')
    plt.title('score_one',fontsize="30")
    plt.savefig(os.path.join(config.plot_path, 'score_one_hidden' + str(config.cluster_num) + '.png'), dpi=300)


from typing import Optional
from scipy.optimize import linear_sum_assignment


def cluster_accuracy(y_true, y_predicted, cluster_number: Optional[int] = None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.
    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy

#
