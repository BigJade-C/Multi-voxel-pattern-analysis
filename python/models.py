# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:59:32 2021

@author: cms
"""

import os
import math
import numpy as np
import pickle
import numpy.matlib as matlab
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE


class MVPA:
    def __init__(self, data, label, mask):
        self.X = data
        self.y = label
        self.mask = mask
        self.normalized = False
        
    def normalize(self, train_idx, test_idx):
        data = self.X
        
        self.tr_ptrn = []
        self.ts_ptrn = []
        
        for vx in range(len(data)):
            tr_target = data[vx][:, train_idx].transpose()
            ts_target = data[vx][:, test_idx].transpose()
            
            mxtr = np.max(tr_target, axis=0); mntr = np.min(tr_target, axis=0)
        
            n_tr_ptrn = np.divide((tr_target-matlab.repmat(mntr, len(tr_target), 1)), matlab.repmat(mxtr-mntr,len(tr_target),1))
            n_ts_ptrn = np.divide((ts_target-matlab.repmat(mntr, len(ts_target), 1)), matlab.repmat(mxtr-mntr,len(ts_target),1))
         
            self.tr_ptrn.append(n_tr_ptrn)    
            self.ts_ptrn.append(n_ts_ptrn)
            
        self.normalized = True

    def zscore(self, train_idx, test_idx):
        data = self.X
        self.tr_ptrn = []
        self.ts_ptrn = []

        for vx in range(len(data)):
            tr_target = data[vx][:, train_idx].transpose()
            ts_target = data[vx][:, test_idx].transpose()

            tr_mean = np.sum(tr_target, axis=0) / len(tr_target)
            tr_difference = [(value - tr_mean) ** 2 for value in tr_target]
            sum_of_difference = np.sum(tr_difference)
            standard_deviation = (sum_of_difference / (len(tr_target) - 1)) ** 0.5

            tr_zscore = [(value - tr_mean) / standard_deviation for value in tr_target]
            ts_zscore = [(value - tr_mean) / standard_deviation for value in ts_target]

            self.tr_ptrn.append(tr_zscore)
            self.ts_ptrn.append(ts_zscore)
        
    def gridSearchCV(self, train_idx, C=None, gamma=None):
        self.v_bestc = []
        self.v_bestg = []
        self.v_bestacc = []

        for voxel_idx in tqdm(range(len(self.X))):
            if self.normalized :
                self.X_train = self.tr_ptrn[voxel_idx]
                self.y_train = self.y[train_idx]
            else:
                self.X_train = self.X[voxel_idx][:, train_idx].transpose()
                self.y_train = self.y[train_idx]

            self.best_acc = 0
            self.best_c = []
            self.best_g = []

            # Set Cost and gamma function parameters
            self.C = [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000,
                      100000000] if C is None else C
            self.gamma = np.logspace(-9, 3, 13) if gamma is None else gamma

            for log2c in self.C:
                for log2g in self.gamma:
                    cv = StratifiedKFold(n_splits=5)
                    cv_accl = []
                    for tr_idx, val_idx in cv.split(self.X_train, self.y_train):
                        model = OneVsRestClassifier(SVC(C=log2c, gamma=log2g), n_jobs=2)
                        model.fit(self.X_train[tr_idx], self.y_train[tr_idx])

                        predict = model.predict(self.X_train[val_idx])
                        cv_acc = accuracy_score(self.y_train[val_idx], predict)
                        cv_accl.append(cv_acc)

                    if np.mean(cv_accl) > self.best_acc:
                        self.best_acc = np.mean(cv_accl)
                        self.best_c = log2c
                        self.best_g = log2g

            self.v_bestc.append(self.best_c)
            self.v_bestg.append(self.best_g)
            self.v_bestacc.append(self.best_acc)

    def test(self, train_idx, test_idx, save_path, C=None, gamma=None):

        self.test_acc = []
        self.test_label = []
        self.cf_matrix = []
        C = self.v_bestc if C is None else [C for v in range(len(self.X))]
        gamma = self.v_bestg if gamma is None else [gamma for v in range(len(self.X))]

        for voxel_idx in tqdm(range(len(self.X))):
            if self.normalized :
                self.X_train = self.tr_ptrn[voxel_idx]
                self.y_train = self.y[train_idx]
                
                self.X_test = self.ts_ptrn[voxel_idx]
                self.y_test = self.y[test_idx]
                
                
            else:
                self.X_train = self.X[voxel_idx][:, train_idx].transpose()
                self.y_train = self.y[train_idx]
    
                self.X_test = self.X[voxel_idx][:, test_idx].transpose()
                self.y_test = self.y[test_idx]

            self.opt_model = OneVsRestClassifier(SVC(C=C[voxel_idx], gamma=gamma[voxel_idx]), n_jobs=2)
            self.opt_model.fit(self.X_train, self.y_train)

            predict = self.opt_model.predict(self.X_test)
            acc = accuracy_score(self.y_test, predict)
            # cf = confusion_matrix(self.y_test, predict)
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            with open(os.path.join(save_path, 'model_%d.pkl' % voxel_idx), 'wb') as f:
                pickle.dump(self.opt_model, f)

            self.test_acc.append(acc)
            self.test_label.append(predict)
            # self.cf_matrix.append(cf)


class linearMVPA:
    def __init__(self, data, label, mask):
        self.X = data
        self.y = label
        self.mask = mask
        self.normalized = False

    def zscore(self, train_idx, test_idx):
        data = self.X
        self.tr_ptrn = []
        self.ts_ptrn = []

        for vx in range(len(data)):
            tr_target = data[vx][:, train_idx].transpose()
            ts_target = data[vx][:, test_idx].transpose()

            tr_mean = np.sum(tr_target, axis=0) / len(tr_target)
            tr_difference = [(value - tr_mean) ** 2 for value in tr_target]
            sum_of_difference = np.sum(tr_difference)
            standard_deviation = (sum_of_difference / (len(tr_target) - 1)) ** 0.5

            tr_zscore = [(value - tr_mean) / standard_deviation for value in tr_target]
            ts_zscore = [(value - tr_mean) / standard_deviation for value in ts_target]

            self.tr_ptrn.append(tr_zscore)
            self.ts_ptrn.append(ts_zscore)

    def normalize(self, train_idx, test_idx):
        data = self.X

        self.tr_ptrn = []
        self.ts_ptrn = []

        for vx in range(len(data)):
            tr_target = data[vx][:, train_idx].transpose()
            ts_target = data[vx][:, test_idx].transpose()

            mxtr = np.max(tr_target, axis=0);
            mntr = np.min(tr_target, axis=0)

            n_tr_ptrn = np.divide((tr_target - matlab.repmat(mntr, len(tr_target), 1)),
                                  matlab.repmat(mxtr - mntr, len(tr_target), 1))
            n_ts_ptrn = np.divide((ts_target - matlab.repmat(mntr, len(ts_target), 1)),
                                  matlab.repmat(mxtr - mntr, len(ts_target), 1))

            self.tr_ptrn.append(n_tr_ptrn)
            self.ts_ptrn.append(n_ts_ptrn)

        self.normalized = True

    def gridSearchCV(self, train_idx, C=None):
        self.v_bestc = []
        self.v_bestg = []
        self.v_bestacc = []

        for voxel_idx in tqdm(range(len(self.X)), desc='Voxel index'):
            if self.normalized:
                self.X_train = self.tr_ptrn[voxel_idx]
                self.y_train = self.y[train_idx]
            else:
                self.X_train = self.X[voxel_idx][:, train_idx].transpose()
                self.y_train = self.y[train_idx]

            self.best_acc = 0
            self.best_c = []
            self.best_g = []

            # Set Cost and gamma function parameters
            self.C = [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000, 10000] if C is None else C


            for log2c in self.C:
                cv = StratifiedKFold(n_splits=5)
                cv_accl = []
                for tr_idx, val_idx in cv.split(self.X_train, self.y_train):
                    model = SGDClassifier(loss='hinge', alpha=log2c, penalty='l2', n_jobs=-1)
                    #model = SVC(C=log2c, kernel='linear')

                    model.fit(self.X_train[tr_idx], self.y_train[tr_idx])

                    predict = model.predict(self.X_train[val_idx])
                    cv_acc = accuracy_score(self.y_train[val_idx], predict)
                    cv_accl.append(cv_acc)

                if np.mean(cv_accl) > self.best_acc:
                    self.best_acc = np.mean(cv_accl)
                    self.best_c = log2c

            self.v_bestc.append(self.best_c)
            self.v_bestg.append(self.best_g)
            self.v_bestacc.append(self.best_acc)

    def test(self, train_idx, test_idx, save_path, C=None):

        self.test_acc = []
        self.test_label = []
        self.cf_matrix = []
        C = self.v_bestc if C is None else [C for v in range(len(self.X))]

        for voxel_idx in tqdm(range(len(self.X))):
            if self.normalized:
                self.X_train = self.tr_ptrn[voxel_idx]
                self.y_train = self.y[train_idx]

                self.X_test = self.ts_ptrn[voxel_idx]
                self.y_test = self.y[test_idx]


            else:
                self.X_train = self.X[voxel_idx][:, train_idx].transpose()
                self.y_train = self.y[train_idx]

                self.X_test = self.X[voxel_idx][:, test_idx].transpose()
                self.y_test = self.y[test_idx]

            self.opt_model = SGDClassifier(loss='hinge', alpha=C[voxel_idx], penalty='l2', n_jobs=-1)
            #self.opt_model = SVC(C=C[voxel_idx], kernel='linear')

            self.opt_model.fit(self.X_train, self.y_train)

            predict = self.opt_model.predict(self.X_test)
            acc = accuracy_score(self.y_test, predict)
            # cf = confusion_matrix(self.y_test, predict)
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            with open(os.path.join(save_path, 'model_%d.pkl' % voxel_idx), 'wb') as f:
                pickle.dump(self.opt_model, f)

            self.test_acc.append(acc)
            self.test_label.append(self.opt_model.coef_)
            # self.cf_matrix.append(cf)


