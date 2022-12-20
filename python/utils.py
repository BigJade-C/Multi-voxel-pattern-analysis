# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:59:55 2021

@author: cms
"""

import os
import numpy as np
from collections import defaultdict
import numpy.matlib as matlab
from sklearn.model_selection import train_test_split

def makeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def tree(): return defaultdict(tree)

def makeLabel(path, tasks, available_sub, mode='avg'):
    label = []
    count = 0
    print("Labeling ,,, ")
    for task in os.listdir(path):
        if task in tasks:
            task_path = os.path.join(path, task)
            for file in os.listdir(task_path):
                if file.endswith('avg.nii') and int(file.split('.')[1].split('_')[0]) in available_sub and mode=='avg':
                    label.append(count)
                elif int(file.split('.')[1].split('_')[0]) in available_sub and mode=='block' and not file.endswith('avg.nii'):
                    label.append(count)
            count = count + 1
    return np.array(label)


def splitData(y, train_size=None, test_size=None):
    shuffle = True

    while shuffle:
        tr_idx, ts_idx = train_test_split(np.arange(len(y)), train_size=train_size, test_size=test_size)
        if len(np.where(y[tr_idx] == 1)[0]) == round(len(tr_idx) * (1 / len(np.unique(y)))):
            shuffle = False

    return tr_idx, ts_idx

def normalize(dataset):
    # Reshape
    if len(dataset.shape) > 2:
        dataset_dim = dataset.shape
        dataset = np.reshape(dataset, (dataset.shape[0], -1))
        
    # Normalize train/test dataset using train data min/max
    mxtr = np.max(dataset, axis=0)
    mntr = np.min(dataset, axis=0)
    
    norm_dataset = np.divide((dataset-matlab.repmat(mntr, len(dataset), 1)), matlab.repmat(mxtr-mntr,len(dataset), 1), where= matlab.repmat(mxtr-mntr,len(dataset), 1)!=0)
    norm_dataset = np.nan_to_num(norm_dataset)

    # Reshape
    r_dataset = np.reshape(norm_dataset, (dataset_dim)) 

    
    return r_dataset

"""
def normalize(trainset, testset):
    # Reshape
    if len(trainset.shape) > 2:
        tr_dim = trainset.shape
        ts_dim = testset.shape
        trainset = np.reshape(trainset, (trainset.shape[0], -1))
        testset = np.reshape(testset, (testset.shape[0], -1))
        
    
    # Normalize train/test dataset using train data min/max
    mxtr = np.max(trainset, axis=0)
    mntr = np.min(testset, axis=0)
    
    norm_trainset = np.divide((trainset-matlab.repmat(mntr, len(trainset), 1)), matlab.repmat(mxtr-mntr,len(trainset), 1))
    norm_testset = np.divide((testset-matlab.repmat(mntr, len(testset), 1)), matlab.repmat(mxtr-mntr,len(testset), 1))
    
    norm_trainset = np.nan_to_num(norm_trainset)
    norm_testset = np.nan_to_num(norm_testset)
    
    # Reshape
    r_trainset = np.reshape(norm_trainset, (tr_dim)) 
    r_testset = np.reshape(norm_testset, (ts_dim)) 
    
    return r_trainset, r_testset
"""