import os
import sys
import h5py
import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_auc_score
from keras.optimizers import Adam

sys.path.insert(0, '../') 
from ecg_AAAI.models.supervised.ecg_fc import build_fc_model
from ecg_AAAI.models.gpu_utils import restrict_GPU_keras
from ecg_AAAI.models.supervised.eval import evaluate_AUC, evaluate_HR, risk_scores
from ecg_AAAI.models.supervised.ablation_helpers import *
from ecg_AAAI.models.supervised.deepsets_model import create_phi, create_rho
restrict_GPU_keras("1")

import warnings
warnings.filterwarnings("error")
warnings.simplefilter("ignore", DeprecationWarning)
split_dir_prefix = "/home/divyas/ecg_AAAI/datasets/split_"
fig_dir = "/home/divyas/ecg_AAAI/models/supervised/figs"
splits = ["0", "1", "2", "3", "4"]
day_threshs = [90, 365, 60, 30]
instances = ['one', 'two',  'three', 'four']
y_modes = ["mi", "cvd"]
n_train_opts = [.1*i for i in range(1, 10)]
train_file = None
test_file = None

# training parameters
block_size = 1500
batch_size = 90
n_results = 10 
n_epochs = 3

# model parameters
model_name = "deepsets"
n_members = 500# first 3 beats
member_dim_opts = (128, 256, 384, 512)
# trimmed parameter space for testing
y_modes = ["cvd"]
result_dicts = []
instances = ['two']
member_dim_opts [256]
day_threshs = [90]
n_train_opts = [.9, .7]
for split_num in splits:
    for y_mode in y_modes:
        for day_thresh in day_threshs:
            for instance_opt, member_dim in zip(instances, member_dim_opts):
                for n_train_opt in n_train_opts:
                    print("\nEXPERIMENT PARAMS:")
                    print("y:", y_mode, "\tday_thresh:", day_thresh, '\t split_num: ', split_num, '\tinstance: ', instance_opt, '\t n_train: ', n_train_opt)
                    if train_file:
                        train_file.close()
                        test_file.close()
                    
                    split_dir = split_dir_prefix + split_num + "/" + instance_opt  

                    train_file = h5py.File(split_dir + "/train.h5", "r")
                    test_file = h5py.File(split_dir + "/test.h5", "r")
                    train_y = get_labels(train_file, y_mode, day_thresh)
                    test_y = get_labels(test_file, y_mode, day_thresh)
                    train_pos_idxs = np.where(train_y == 1)[0]
                    train_pos_idxs = np.random.choice(train_pos_idxs, int(n_train_opt*len(train_pos_idxs)), replace=False)
                    train_pos_idxs = sorted(train_pos_idxs)

                    x_train_pos = train_file['adjacent_beats'][list(train_pos_idxs),:n_members]
                    y_train_pos = train_file[y_mode + '_labels'][list(train_pos_idxs)]
                    y_train_pos = thresh_labels(y_train_pos, day_thresh)

                    batch_size = block_size 
                    n_batches = int(block_size/batch_size + 1)
                    n_blocks = int(len(train_y)/block_size + 1)
                    print("N blocks: ", n_blocks)
               
                    set_dims = (n_members, member_dim)
                    member_dims = (member_dim,1)
                    phi = create_phi(member_dims)
                    rho = create_rho(member_dims, phi, n_members)
                    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                    rho.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
                    for i in range(n_results):
                        for j in range(n_blocks):
                            # Load 1000 patients into memory at a time
                            x_train_block, y_train_block = get_block(train_file, j, block_size, 
                                                                     y_mode, day_thresh, n_beats = n_members)
                            print("Finished loading Block #", j)
                            x_train_batch = np.concatenate([x_train_block, x_train_pos])
                            y_train_batch = np.concatenate([y_train_block, y_train_pos])
                            swapped = np.swapaxes(x_train_batch, 0, 1)
                            
                            if swapped.shape[1] > len(y_train_batch):
                                swapped = swapped[:,:-1,:]
                            swapped = np.expand_dims(swapped, 3)
                            rho_input = [x for x in swapped]
                            cw = get_class_weights(y_train_batch, .1)

                            rho.fit(x=rho_input, y=y_train_batch, epochs=n_epochs, verbose=False, batch_size=80000, class_weight=cw)
                        
                        py_pred = get_preds_deepsets(rho, test_file, n_members)
                        if len(test_y) != len(py_pred):
                            fixed_length = min(len(test_y), len(py_pred))
                            test_y = test_y[:fixed_length]
                            py_pred = py_pred[:fixed_length]

                        auc_score = roc_auc_score(test_y, py_pred)
                        true_y = test_file[y_mode + "_labels"][:]
                        hr_score = 0
                        discrete_hr = 0
                        or_score = 0
                        try:
                            or_score = calc_or(test_y, py_pred)
                        except:
                            print("Error calculating HR")
                        print("AUC score: ", auc_score)
                        print("HR: ", hr_score)

                        result_dict = {'y_mode': y_mode, 'epoch': i, 'model': model_name, 
                                           'pauc': auc_score, 'hr': hr_score, 'day_thresh': day_thresh, 'instance': instance_opt, 'split_num': split_num, 'discrete_hr': discrete_hr, 'or': or_score, 'n_train': n_train_opt}
                        result_dicts.append(result_dict)
                        pd.DataFrame(result_dicts).to_csv("deepsets_df_all")
