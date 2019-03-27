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
restrict_GPU_keras("0")

split_dir_prefix = "/home/divyas/ecg_AAAI/datasets/splits/split_"
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
n_results = 4 
n_epochs = 2

# model parameters
model_name = "deepsets"
n_members = 500# first 3 beats
member_dim_opts = (128, 256, 384, 512)
# trimmed parameter space for testing
y_modes = ["cvd"]
result_dicts = []
instances = ['two']
member_dim_opts = [256]
splits = ["1", "2", "3", "4"]
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
                    x_train_pos = train_file['adjacent_beats'][list(train_pos_idxs),:n_members]
                    y_train_pos = train_file[y_mode + '_labels'][list(train_pos_idxs)]
                    y_train_pos = thresh_labels(y_train_pos, day_thresh)
                    n_pos = len(x_train_pos)
                    keep_idxs = np.random.choice(n_pos, int(n_train_opt*n_pos))
                    x_train_pos, y_train_pos = x_train_pos[keep_idxs], y_train_pos[keep_idxs]
                    n_train_pos = int(n_train_opt*n_pos) 
                    
                    batch_size = n_train_pos
                    n_batches = int(block_size/batch_size + 1)
                    n_blocks = int(len(train_y)/block_size + 1)
                    print("N blocks: ", n_blocks)
               
                    set_dims = (n_members, member_dim)
                    member_dims = (member_dim,1)
                    phi = create_phi(member_dims)
                    rho = create_rho(member_dims, phi, n_members)
                    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                    rho.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
                    keep_idxs = np.random.choice(block_size, int(n_train_opt*block_size))
                    for i in range(n_results):
                        for j in range(n_blocks):
                            # Load 1000 patients into memory at a time
                            x_train_block, y_train_block = get_block(train_file, j, block_size, 
                                                                     y_mode, day_thresh, n_beats = n_members)
                            print("Finished loading Block #", j)
                            filt_idxs = [x for x in keep_idxs if x < x_train_block.shape[0] - 1]
                            x_train_block, y_train_block = x_train_block[filt_idxs], y_train_block[filt_idxs]
                            n_batches = int(x_train_block.shape[0]/batch_size + 1) 
                            print("N batches: ", n_batches)
                            for k in range(n_batches):
                                x_train_neg, y_train_neg = get_block_batch_deepsets(x_train_block, y_train_block, 
                                                                           batch_size, k, n_beats=n_members) 
                                x_train_batch = np.concatenate([x_train_neg, x_train_pos])
                                y_train_batch = np.concatenate([y_train_neg, y_train_pos])
                                swapped = np.swapaxes(x_train_batch, 0, 1)
                                swapped = np.expand_dims(swapped, 3)
                                rho_input = [x for x in swapped]
                                rho.fit(x=rho_input, y=y_train_batch, epochs=n_epochs, 
                                      verbose=False, batch_size=80000)
                        
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
                        rho.save("./deepsets/rho_" + y_mode + "_epoch_" + str(int(i)) + "_" + str(int(day_thresh)) + "_" + str(int(split_num)) + "_" +  model_name + ".h5" )

                        result_dict = {'y_mode': y_mode, 'epoch': i, 'model': model_name, 
                                           'pauc': auc_score, 'hr': hr_score, 'day_thresh': day_thresh, 'instance': instance_opt,
                                           'split_num': split_num, 'discrete_hr': discrete_hr, 'or': or_score, 'n_train': n_train_opts}
                        result_dicts.append(result_dict)
                        pd.DataFrame(result_dicts).to_csv("deepsets_n_train_df_2")
