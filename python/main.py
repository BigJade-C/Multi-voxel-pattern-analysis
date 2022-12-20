# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:59:19 2021

@author: cms
"""
import sys
import os
from tqdm import tqdm
import datetime as dt
import pytz
from multiprocessing import Process

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))

from MVPA import models
from MVPA import analysis
from MVPA import loads
from MVPA import utils
from MVPA import save

import numpy as np

def main(task):
#
	KST = pytz.timezone('Asia/Seoul')
	x = dt.datetime.now(KST)
	date = x.strftime("%Y%m%d")

	smooth = '0'
	#task = ['Rknee', 'Rtoe']
	radius = 1
	taskname = [task[i][:2].upper() for i in range(len(task))]
	taskname = ''.join(taskname)
	n_perm = 100
	available_sub = [1, 2, 7, 9, 10, 11, 13, 15, 16, 17, 18, 20, 21, 22, 23, 26, 27, 30, 31, 32, 36]

	# Define paths
	path = '/users/cms/data/stroke/input/fwhm%s' % smooth
	save_path = '/users/cms/data/NR/searchlight-MVPA/' + date + '_perm_beta_sm' + smooth + '_rds' + str(radius)
	ROI_path = '/users/cms/data/stroke/new_blockWise/mask/PCL_intergroup.nii'

	# Load data
	data, f_names = loads.loadData(path, task, available_sub)
	mask, affine = loads.loadMask(ROI_path)
	labels = utils.makeLabel(path, task, available_sub)
	print("====== Finish Load Data ,,, ======")

	# Check exists save directory
	utils.makeDir(save_path)
	utils.makeDir(os.path.join(save_path, 'models'))
	utils.makeDir(os.path.join(save_path, 'nii'))
	utils.makeDir(os.path.join(save_path, 'npz'))

	# define sphere size and extract pattern of each sphere
	transpose_data = np.transpose(data, (1,2,3,0))
	sphere = analysis.Searchlight(radius)
	pattern = [inform for inform in sphere.analysis(transpose_data, mask)]
	print("====== Finish Make Searchlight analysis information ,,, ======")

	betas = []; coordinates = []
	for v, c in tqdm(pattern):
		betas.append(v)
		coordinates.append(c)

	# get index
	print("====== Make Train/Test Index ,,, ======")
	train_idx, test_idx = utils.splitData(labels)

	# MVPA
	print("====== Start Model Training ,,, ======")
	perm_acc = []
	perm_mean = []
	perm_std = []
	for perm in tqdm(range(n_perm)):
		mvpa = models.linearMVPA(betas, labels, mask)

		mvpa.normalize(train_idx, test_idx)
		# GridSearch
		mvpa.gridSearchCV(train_idx)

		# Test
		mvpa.test(train_idx, test_idx, os.path.join(save_path, 'models', taskname))

		# Save
		perm_acc.append(mvpa.test_acc)
		perm_coef = mvpa.test_label

		#save.saveNifti(mvpa.test_acc, mvpa.v_bestacc, mvpa.test_label, mask, coordinates, affine, os.path.join(save_path, 'nii'), taskname+'_%s.nii' % str(perm))
		#save.saveInfo(mvpa, train_idx, test_idx, radius, mask, coordinates, affine, os.path.join(save_path,'npz'), taskname+'_info_%s.npz' % str(perm), betas, f_names)
		mean = []
		std = []
		for idx in range(len(perm_coef)):
			mean.append(np.mean(perm_coef[idx][0]))
			std.append(np.std(perm_coef[idx][0]))

		perm_mean.append(mean)
		perm_std.append(std)
	np.savez(os.path.join(save_path, '%s_permutation.npz' % (taskname)), acc=perm_acc, mean=perm_mean, std=perm_std)

if __name__=='__main__':
	task_arr = [
				['Lflex', 'Lknee'],
				['Lflex', 'Ltoe'],
				['Lflex', 'Lrot'],
				['Lknee', 'Ltoe'],
				['Lknee', 'Lrot'],
				['Ltoe', 'Lrot'],
				['Rflex', 'Rknee'],
				['Rflex', 'Rtoe'],
				['Rflex', 'Rrot'],
				['Rknee', 'Rtoe'],
				['Rknee', 'Rrot'],
				['Rtoe', 'Rrot'],
				['Lflex', 'Rflex'],
				['Lknee', 'Rknee'],
				['Ltoe', 'Rtoe'],
				['Lrot', 'Rrot']
				]

	procs = []

	for task in task_arr:
		proc = Process(target=main, args=(task, ))
		procs.append(proc)
		proc.start()

	for proc in procs:
		proc.join()
