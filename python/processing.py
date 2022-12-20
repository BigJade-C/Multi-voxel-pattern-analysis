# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:59:19 2021

@author: cms
"""
from multiprocessing import Process
from tqdm import tqdm
import time
import os

import models
import analysis
import loads
import utils
import save

import numpy as np

def main(task):
    # Define paths 
    path = '/users/cms/data/stroke/new_blockWise/fwhm0'
    save_path = '/users/cms/data/NR/searchlight-MVPA/2112161915_rds1'
    ROI_path = '/users/cms/data/stroke/new_blockWise/mask/PCL_intergroup.nii'
    
    radius = 1
    taskname = [task[i][:2].upper() for i in range(len(task))]
    taskname = ''.join(taskname)

    print("Radius size of Sphere : %d" % radius)

    # Load data
    data, f_names = loads.loadData(path, task, 'BLOCK')
    mask, affine = loads.loadMask(ROI_path)
    labels = utils.makeLabel(path, task, 'BLOCK')
    
    # define sphere size and extract pattern of each sphere 
    transpose_data = np.transpose(data, (1,2,3,0))
    sphere = analysis.Searchlight(radius)
    pattern = [inform for inform in sphere.analysis(transpose_data, mask)]
    
    betas = []; coordinates = []
    for v, c in pattern:
    	betas.append(v)
    	coordinates.append(c)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'npz'))
        os.mkdir(os.path.join(save_path, 'models'))
        os.mkdir(os.path.join(save_path, 'nii'))

    # get index 
    train_idx, test_idx = utils.splitData(labels)
    
    # MVPA 
    mvpa = models.MVPA(betas, labels, mask)
    mvpa.normalize(train_idx, test_idx)

    # GridSearch
    mvpa.gridSearchCV(train_idx)
    
    # Test
    mvpa.test(train_idx, test_idx, os.path.join(save_path, 'models', taskname))
    
    # Save
    save.saveNifti(mvpa.test_acc, mvpa.v_bestacc, mvpa.test_label, mask, coordinates, affine, os.path.join(save_path, 'nii'), taskname+'.nii')
    save.saveInfo(mvpa, train_idx, test_idx, radius, mask, coordinates, affine, os.path.join(save_path, 'npz'), taskname+'_info.npz', betas, f_names)
    
if __name__ == '__main__':
    task_arr = [['Lflex','Lknee'],
                        ['Lflex','Ltoe'],
                        ['Lflex','Lrot'],
                        ['Lknee','Ltoe'],
                        ['Lknee','Lrot'],
                        ['Ltoe','Lrot'],
                        ['Rflex','Rknee'],
                        ['Rflex','Rtoe'],
                        ['Rflex','Rrot'],
                        ['Rknee','Rtoe'],
                        ['Rknee','Rrot'],
                        ['Rtoe','Rrot'],
                        ['Lflex','Rflex'],
                        ['Lknee','Rknee'],
                        ['Ltoe','Rtoe'],
                        ['Lrot','Rrot'] ]
    """
            ['Lflex', 'Lknee', 'Ltoe', 'Lrot']]
                       #['Rflex', 'Rknee', 'Rtoe', 'Rrot']]              
    """
    
    procs = []
    
    for index, task in tqdm(enumerate(task_arr)):
        proc = Process(target=main, args=(task, ))
        procs.append(proc)
        proc.start()
        
    for proc in procs:
        proc.join()
    