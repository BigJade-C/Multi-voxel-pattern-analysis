# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:00:08 2021

@author: cms
"""

import os

import nibabel as nib
import numpy as np
    

def loadData(path, tasks, available_sub, mode='avg'):
    data = []
    filename = []
    for task in sorted(os.listdir(path)):
        if task in tasks:
            task_path = os.path.join(path, task)
            for file in sorted(os.listdir(task_path)):
                if file.endswith('avg.nii') and int(file.split('.')[1].split('_')[0]) in available_sub and mode=='avg':
                    filename.append(file)
                    img = nib.load(os.path.join(task_path, file))
                    nii = img.get_fdata()
                    data.append(nii)
                elif int(file.split('.')[1].split('_')[0]) in available_sub and mode=='block' and not file.endswith('avg.nii') :
                    filename.append(file)
                    img = nib.load(os.path.join(task_path, file))
                    nii = img.get_fdata()
                    data.append(nii)

    data = np.array(data)
    return data, filename

def loadMask(path):
    img = nib.load(path)
    nii = img.get_fdata()
    affine = img.affine
    inter_mask = np.array(nii)

    return inter_mask, affine