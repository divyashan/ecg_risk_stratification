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
from ecg_AAAI.models.ecg_utils import get_all_adjacent_beats
from ecg_AAAI.models.supervised.ecg_fi_model_keras import build_fi_model 
from ecg_AAAI.models.supervised.ecg_fc import build_fc_model
from ecg_AAAI.models.gpu_utils import restrict_GPU_keras
from ecg_AAAI.models.supervised.eval import evaluate_AUC, evaluate_HR, risk_scores
import tftables
restrict_GPU_keras("3")

y_modes = ["mi", "cvd"]
splits = ["0", "1", "2", "3", "4"]
day_threshs = [30, 60, 90, 365]

model_name = "fc"
fig_dir = "/home/divyas/ecg_AAAI/models/supervised/figs"
batch_size = 80
day_thresh = 90
input_dim = 256
train_file = None
test_file = None
n_iters = 41


# Set up directory structure in case it's not there
def get_fig_path(y_mode, day_thresh, split_num):
    return fig_dir + "/" + y_mode + "/" + str(day_thresh) + "/split_" + split_num

def get_day_path(y_mode, day_thresh):
    return fig_dir + "/" + y_mode + "/" + str(day_thresh)


def get_batch(fhandle, batch_i, bs):
    start = batch_i*bs
    end = start + bs
    
    x_train = fhandle['adjacent_beats'][start:end]
    x_train = np.reshape(x_train, (x_train.shape[0]*x_train.shape[1], x_train.shape[2]))
    x_train = np.expand_dims(x_train, 2)
    
    y_train = fhandle[y_mode + "_labels"][start:end]
    y_train = [[1]*3600 if (y_val < day_thresh and y_val > 0) else [0]*3600  for y_val in y_train]
    y_train = np.array(y_train).flatten()
    
    return x_train, y_train

def get_preds(m, y_mode, test_file):
    iy_pred = []
    py_pred = []
    n_batches = test_file['adjacent_beats'].shape[0]
    for i in range(n_batches):
        x_test_batch, _ = get_batch(test_file, i, 1)
        extension = m.predict(x_test_batch)
        #iy_pred.extend(extension)
        py_pred.append(np.mean(extension))
    all_y = test_file[y_mode + "_labels"][:]
    zero_idxs = np.concatenate([np.where(all_y == 0)[0], np.where(all_y > day_thresh)[0]])
    binary_y = np.ones(all_y.shape)
    binary_y[zero_idxs] = 0
    return np.array(iy_pred), np.array(py_pred), binary_y

for y_mode in y_modes:
    for split_num in splits:
        for day_thresh in day_threshs:
            if not os.path.isdir(get_fig_path(y_mode, day_thresh, split_num)):
                if not os.path.isdir(get_day_path(y_mode, day_thresh)):
                    os.mkdir(get_day_path(y_mode, day_thresh))
                os.mkdir(get_fig_path(y_mode, day_thresh, split_num))

m, embedding_m = build_fc_model((input_dim, 1))
m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
class_weight = {0: 1., 1: 50.}

result_dicts = []
for y_mode in y_modes:
    for day_thresh in day_threshs:
        if y_mode == "mi" and day_thresh == 30:
            continue
        for split_num in splits:
            
            print("\nEXPERIMENT PARAMS:")
            print("y:", y_mode, "\tday_thresh:", day_thresh, '\t split_num: ', split_num)
            if split_num == "1":
                continue
            if train_file:
                train_file.close()
                test_file.close()
            split_dir = "/home/divyas/ecg_AAAI/datasets/splits/split_" + split_num
            train_file = h5py.File(split_dir + "/train.h5", "r")
            test_file = h5py.File(split_dir + "/test.h5", "r")
            n_train_patients = int(train_file['adjacent_beats'].shape[0])
            n_train_batches = int(n_train_patients/batch_size)
            n_test_patients = int(test_file['adjacent_beats'].shape[0])
            n_test_batches = int(n_test_patients/batch_size)
            for i in range(n_iters):
                for j in range(n_train_batches):
                    x_train_batch, y_train_batch = get_batch(train_file, j, batch_size)
                    m.fit(x=x_train_batch, y=y_train_batch, epochs=1, 
                          verbose=False, batch_size=60000, class_weight=class_weight)

                # Plotting
                if i % 5 == 0 and i != 0:
                    iy_pred, py_pred, all_y = get_preds(m, y_mode, test_file)
                    if len(all_y) != len(py_pred):
                        fixed_length = min(len(all_y), len(py_pred))
                        all_y = all_y[:fixed_length]
                        py_pred = py_pred[:fixed_length]

                    auc_score = roc_auc_score(all_y, py_pred)
                    fig_path = get_fig_path(y_mode, day_thresh, split_num)

                    plt.hist(py_pred[np.where(all_y == 1)], color='red', alpha=.5, bins=20)
                    plt.title("[" + y_mode
                              +  " positive] distribution of risk scores (90 days) AUC = "
                              + str(auc_score))
                    plt.xlim(0, 1)
                    plt.savefig(fig_path +"/epoch_" + str(i) + "_positive")
                    plt.clf()

                    plt.hist(py_pred[np.where(all_y != 1)], color='green', alpha=.5, bins=20)
                    plt.title("[" + y_mode +  " negative] distribution" + 
                              "of risk scores (90 days) AUC = " + str(auc_score))
                    plt.xlim(0, 1)
                    plt.savefig(fig_path +"/epoch_" + str(i) + "_negative")
                    plt.clf()
                    
                    result_dict = {'y_mode': y_mode, 'epoch': i, 'model': model_name, 
                                   'pauc': auc_score,'day_thresh': day_thresh, 
                                   'split_num': split_num}
                    result_dicts.append(result_dict)
                    pd.DataFrame(result_dicts).to_csv("results_df")




        
