# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:05:56 2021

@author: cms
"""

import numpy as np
import os
import nibabel as nib

def saveNifti(arr, cv_arr, label, mask, coordinates, affine, path, filename):
    arr = np.array(arr)
    arr_cv = np.array(cv_arr)
    #label = np.array(label)
    coordinates = coordinates
    affine = affine
    path= path
    dim = mask.shape
    filename = filename
    
    cv_arr = np.zeros((dim))
    test_arr = np.zeros((dim))
    label_arr = np.zeros((dim))
    label_arr2 = np.zeros((dim))
    
    for v_idx in range(len(arr_cv)):
        cv_arr[coordinates[v_idx]] = arr_cv[v_idx]
    
    for v_idx in range(len(arr)):
        test_arr[coordinates[v_idx]] = arr[v_idx]
    
    for v_idx in range(len(label)):
        print(len(label), len(label[v_idx][0]))
        v_label = np.mean(label[v_idx][0])
        v_label2 = np.std(label[v_idx][0])
        label_arr[coordinates[v_idx]] = v_label
        label_arr2[coordinates[v_idx]] = v_label2

        #v_label = label[v_idx][1]
        #label_Arr2[coordinates[v_idx]] = v_label

        #label_arr[coordinates[v_idx]] = sum(v_label) / label.shape[1] - 0.5
        #v_label = np.mean(v_label)
        #print(v_label.shape)
        #print(label_arr.shape)

        #label_arr = np.mean(np.array(label_arr)[:, 0, :], axis=1)
    
    cv_img = nib.Nifti1Image(cv_arr, affine=affine)
    test_img = nib.Nifti1Image(test_arr, affine=affine)
    label_img = nib.Nifti1Image(label_arr, affine=affine)
    label2_img = nib.Nifti1Image(label_arr2, affine=affine)
    
    nib.save(cv_img, os.path.join(path, 'cvacc_'+filename))
    nib.save(label_img, os.path.join(path, 'mean_'+filename))
    nib.save(label2_img, os.path.join(path, 'std_' + filename))
    nib.save(test_img, os.path.join(path, 'tsacc_'+filename))
        

    
    
def saveInfo(model, train_idx, test_idx, radius, mask, coordinates, affine, path, filename, betas, data_path, filenames=None):
    coordinates = coordinates
    mask = mask
    affine = affine
    path = path
    filename = filename
    radius = radius
    acc = model.test_acc
    #label = model.test_label
    data_path = data_path
    betas = betas
    
    train_idx = train_idx
    test_idx = test_idx
    
    np.savez(os.path.join(path, filename), coordinates=coordinates, affine=affine, data_path=data_path,
             mask=mask, radius=radius, acc=acc, train_idx=train_idx, test_idx=test_idx, betas=betas, filenames=filenames)
    
    """
    info = utils.tree()
    
    info['coordinates'] = coordinates
    info['mask'] = mask
    info['radius'] = radius
    info['acc'] = acc
    info['label'] = label
    info['train_idx'] = train_idx
    info['test_idx'] = test_idx
    
    df = pd.DataFrame_from_dict(info)
    """