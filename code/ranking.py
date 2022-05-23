# %%
# Importing libraries
import pandas as pd
import numpy as np
import random

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from lambdaMART import LambdaMART

import warnings
warnings.filterwarnings("ignore")
# %%
# Defining Functions and ignoring warnings
def dcg(rel, k=None):
    i = np.arange(1, len(rel)+1)
    gain = (2**rel - 1)/np.log2(i + 1)
    if k is not None:
        gain = gain[i <= k]
    return gain.sum()

def idcg(rel, k=None):
    rel = np.sort(rel)[::-1]
    i = np.arange(1, len(rel)+1)
    gain = (2**rel - 1)/np.log2(i + 1)
    if k is not None:
        gain = gain[i <= k]
    return gain.sum()

def ndcg(rel, k=None):
    idcg_value = idcg(rel, k=k)
    if idcg_value != 0:
        return dcg(rel, k=k) / idcg_value
    else:
        return 0

def ndcg_mean(res_table, k=None):
    ndcg_val = 0
    for qid in res_table['QueryId'].unique():
        rel = res_table[res_table['QueryId'] == qid]['rel']
        ndcg_val += ndcg(rel, k=k)
    return ndcg_val / res_table['QueryId'].nunique()

# %%
# Defining testing and training data
train_datafile = '../Dataset/VU_DM_data/fixed_training_set_VU_DM.csv'
test_datafile = '../Dataset/VU_DM_data/filled_test_set_VU_DM.csv'

df_train, pos_train, srch_id_train = load_svmlight_file(train_datafile, query_id = True)


# %%
