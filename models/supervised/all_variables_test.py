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
from ecg_AAAI.models.gpu_utils import restrict_GPU_keras
from ecg_AAAI.models.supervised.ablation_helpers import *
restrict_GPU_keras("0")

y_modes = ["mi", "cvd"]
splits = ["0", "1", "2", "3", "4"]
day_threshs = [30, 60, 90, 365]
pred_fs = [np.mean, np.median, top_10, top_20]
pred_f_names = ['mean', 'median', 'top_10', 'top_20']
fig_dir = "/home/divyas/ecg_AAAI/models/supervised/figs"

n_fc_units = 1
model_name = "fc" + str(int(n_fc_units))
batch_size = 80
day_thresh = 90
input_dim = 256
n_epochs = 5
block_size = 1500
n_results = 3

train_file = None
test_file = None
plotting = False

# # Set up directory structure in case it's not there
# for y_mode in y_modes:
#     for split_num in splits:
#         for day_thresh in day_threshs:
#             if not os.path.isdir(get_fig_path(y_mode, day_thresh, split_num, model_name)):
#                 if not os.path.isdir(get_day_path(y_mode, day_thresh)):
#                     os.mkdir(get_day_path(y_mode, day_thresh))
#                 if not os.path.isdir(get_split_path(y_mode, day_thresh, split_num)):
#                     os.mkdir(get_split_path(y_mode, day_thresh, split_num))
#                 os.mkdir(get_fig_path(y_mode, day_thresh, split_num, model_name))

splits = ["0"]
day_threshs = [90]
result_dicts = []
for y_mode in y_modes:
    for day_thresh in day_threshs:
        for split_num in splits:
            print("\nEXPERIMENT PARAMS:")
            print("y:", y_mode, "\tday_thresh:", day_thresh, '\t split_num: ', split_num)
            if train_file:
                train_file.close()
                test_file.close()
            m, embedding_m = build_fc_model((input_dim, 1))
            m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            split_dir = "/home/divyas/ecg_AAAI/datasets/splits/split_" + split_num
            train_file = h5py.File(split_dir + "/train.h5", "r")
            test_file = h5py.File(split_dir + "/test.h5", "r")


            train_y = get_labels(train_file, y_mode, day_thresh)
            test_y = get_labels(test_file, y_mode, day_thresh)
            train_pos_idxs = np.where(train_y == 1)[0]
            x_train_pos = train_file['adjacent_beats'][list(train_pos_idxs)]
            x_train_pos = reshape_X(x_train_pos)
            y_train_pos = train_file[y_mode + '_labels'][list(train_pos_idxs)]
            y_train_pos = thresh_labels(y_train_pos, day_thresh)
            y_train_pos = np.array([[y_val]*3600 for y_val in y_train_pos]).flatten()

            n_train_pos = len(train_pos_idxs)
            batch_size = n_train_pos
            n_batches = int(block_size/batch_size + 1)
            n_blocks = int(len(train_y)/block_size + 1)

            for i in range(n_results):
                for j in range(n_blocks):
                    x_train_block, y_train_block = get_block(train_file, i, block_size, y_mode, day_thresh)
                    print("Finished loading Block #", j)
                    for k in range(n_batches):
                        x_train_neg, y_train_neg = get_block_batch(x_train_block, y_train_block, batch_size, k) 
                        print("Done getting batch #", k)

                        x_train_batch = np.concatenate([x_train_neg, x_train_pos])
                        y_train_batch = np.concatenate([y_train_neg, y_train_pos])
                        m.fit(x=x_train_batch, y=y_train_batch, epochs=n_epochs, verbose=False, batch_size=80000)
                
                # TODO: test all functions w/o regenerating instance predictions
                for pred_f, pred_f_name in zip(pred_fs, pred_f_names):
                    py_pred = get_preds(m, test_file, pred_f)
                    if len(test_y) != len(py_pred):
                        fixed_length = min(len(test_y), len(py_pred))
                        test_y = test_y[:fixed_length]
                        py_pred = py_pred[:fixed_length]

                    auc_score = roc_auc_score(test_y, py_pred)
                    hr_score = calc_hr(test_y, py_pred)
                    m.save("m_" + y_mode + "_epoch_" + str(int(i)) + "_" + str(int(day_thresh)) + "_" + str(int(split_num)) + "_" +  model_name + ".h5" )
                    embedding_m.save("m_" + y_mode + "_epoch_" + str(int(i)) + "_" + str(int(day_thresh)) + "_" + str(int(split_num)) + "_" +  model_name + ".h5")

                    result_dict = {'y_mode': y_mode, 'epoch': i, 'model': model_name, 
                                   'pauc': auc_score,'day_thresh': day_thresh, 'pred_f': pred_f_name,
                                   'split_num': split_num}
                    result_dicts.append(result_dict)
                    pd.DataFrame(result_dicts).to_csv("restructured_training_df_4")
                    
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
                    



        
