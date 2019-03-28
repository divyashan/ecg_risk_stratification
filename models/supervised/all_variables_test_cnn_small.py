import os
import sys
import h5py
import pdb
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat

sys.path.insert(0, '../')
from ecg_AAAI.parse_dataset.readECG import loadECG
from ecg_AAAI.models.supervised.ecg_fi_model_keras import build_fi_model 
from ecg_AAAI.models.supervised.ecg_fc import build_fc_model
from ecg_AAAI.models.supervised.ecg_cnn import build_cnn, build_small_f_cnn
from ecg_AAAI.models.gpu_utils import restrict_GPU_keras
from ecg_AAAI.models.supervised.ablation_helpers import *
restrict_GPU_keras("0")
import warnings
warnings.filterwarnings("error")
warnings.simplefilter("ignore", DeprecationWarning)

y_modes = ["mi", "cvd"]
splits = ["4", "3", "0", "1", "2"]
day_threshs = [365, 90, 30, 60]
pred_fs = [np.mean, np.median, top_10_mean, top_20_mean]
pred_f_names = ['mean', 'median', 'top_10_mean', 'top_20_mean']
instances = ['four', 'one', 'two', 'three', 'four']
split_prefix = "/home/divyas/ecg_AAAI/datasets/split_"
fig_dir = "/home/divyas/ecg_AAAI/models/supervised/figs"

model_name = "small_filter_cnn"
batch_size = 90
day_thresh = 90
n_epochs = 1
block_size = 1500
n_results = 4 
n_beats = 500 

train_file = None
test_file = None
plotting = False

pred_fs = [top_20_mean]
pred_f_names = ['top_20']
y_modes = ['cvd']
result_dicts = []
for split_num in splits:
    for y_mode in y_modes:
        for day_thresh in day_threshs:
            for instance_opt in instances:
                print("\nEXPERIMENT PARAMS:")
                print("y:", y_mode, "\tday_thresh:", day_thresh, '\t split_num: ', split_num, '\t instance: ', instance_opt)
                if train_file:
                    train_file.close()
                    test_file.close()

                split_dir = split_prefix + split_num + "/" + instance_opt 
                train_file = h5py.File(split_dir + "/train.h5", "r")
                test_file = h5py.File(split_dir + "/test.h5", "r")


                train_y = get_labels(train_file, y_mode, day_thresh)
                test_y = get_labels(test_file, y_mode, day_thresh)
                train_pos_idxs = np.where(train_y == 1)[0]
                x_train_pos = train_file['adjacent_beats'][list(train_pos_idxs),:n_beats,:]
                x_train_pos = reshape_X(x_train_pos)
                y_train_pos = train_file[y_mode + '_labels'][list(train_pos_idxs)]
                y_train_pos = thresh_labels(y_train_pos, day_thresh)
                y_train_pos = np.array([[y_val]*n_beats for y_val in y_train_pos]).flatten()

                n_train_pos = len(train_pos_idxs)
                batch_size = n_train_pos
                n_batches = int(block_size/batch_size + 1)
                n_blocks = int(len(train_y)/block_size + 1)
                print("N blocks: ", n_blocks)
                print("N batches: ", n_batches)
                print("Batch size: ", batch_size)
                
                input_dim = x_train_pos.shape[-2]
                m = build_small_f_cnn((input_dim, 1))
                m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
                for i in range(n_results):
                    for j in range(n_blocks):
                        x_train_block, y_train_block = get_block(train_file, j, block_size, y_mode, day_thresh, n_beats=n_beats)
                        print("Finished loading Block #", j)
                        for k in range(n_batches):
                            x_train_neg, y_train_neg = get_block_batch(x_train_block, y_train_block, batch_size, k, n_beats=n_beats) 

                            x_train_batch = np.concatenate([x_train_neg, x_train_pos])
                            y_train_batch = np.concatenate([y_train_neg, y_train_pos])
                            m.fit(x=x_train_batch, y=y_train_batch, epochs=n_epochs, verbose=False,  batch_size=10000)
                
                    # TODO: test all functions w/o regenerating instance predictions
                    for pred_f, pred_f_name in zip(pred_fs, pred_f_names):
                        print("Starting pred_f: ", pred_f_name)
                        py_pred = get_preds(m, test_file, pred_f, n_beats)
                        
                        if len(test_y) != len(py_pred):
                            fixed_length = min(len(test_y), len(py_pred))
                            test_y = test_y[:fixed_length]
                            py_pred = py_pred[:fixed_length]

                        auc_score = roc_auc_score(test_y, py_pred)
                        print("AUC: ", auc_score)
                        true_y = test_file[y_mode + "_labels"][:]
                        hr_score = 0
                        discrete_hr = 0
                        try:
                            hr_score = calc_hr(true_y, py_pred)
                            discrete_hr = calc_hr(true_y, py_pred, discretize=True)
                            or_score = calc_or(test_y, py_pred)
                        except:
                            print("Error calculating HR")
                        print("HR: ", hr_score)
                        print("Discrete HR: ", discrete_hr)
                        result_dict = {'y_mode': y_mode, 'epoch': i, 'model': model_name, 'instance': instance_opt, 
                               'pauc': auc_score, 'hr': hr_score, 'day_thresh': day_thresh, 'pred_f': pred_f_name,
                               'split_num': split_num, 'discrete_hr': discrete_hr, 'or': or_score}
                        result_dicts.append(result_dict)
                        pd.DataFrame(result_dicts).to_csv("small_cnn_all_parameters_df_with_or")
                        
                        if plotting:
                            fig_path = get_fig_path(y_mode, day_thresh, split_num, model_name)
                            plt.xlim(0, 1)
                            plt.hist(py_pred[np.where(all_y == 1)], color='red', alpha=.5, bins=20)
                            plt.title("[" + y_mode +  " positive] distribution of risk scores (90 days) AUC = " + str(auc_score))
                            plt.savefig(fig_path +"/epoch_" + str(i) + "_positive")
                            plt.clf()

                            plt.xlim(0, 1)
                            plt.hist(py_pred[np.where(all_y != 1)], color='green', alpha=.5, bins=20)
                            plt.title("[" + y_mode +  " negative] distribution" + "of risk scores (90 days) AUC = " + str(auc_score))
                            plt.savefig(fig_path +"/epoch_" + str(i) + "_negative")
                            plt.clf()
                            



                    
