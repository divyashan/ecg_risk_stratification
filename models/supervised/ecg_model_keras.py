import os
import sys
import math
import h5py
import numpy as np
import pdb


from keras.models import Model
from keras.layers import Input, Flatten, Dense, Reshape, Lambda, Add
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv1D, MaxPooling1D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.initializers import glorot_normal
from keras.regularizers import l2
import keras.backend as K

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '../')
from ecg_AAAI.parse_dataset.readECG import loadECG
from ecg_AAAI.models.ecg_utils import get_all_adjacent_beats
from ecg_AAAI.models.supervised.ecg_fi_model_keras import build_fi_model 
from ecg_AAAI.models.supervised.ecg_fc import build_fc_model
from ecg_AAAI.models.gpu_utils import restrict_GPU_keras
from ecg_AAAI.models.supervised.eval import evaluate_AUC, evaluate_HR, risk_scores
restrict_GPU_keras("0")

"""Hyperparameters"""
max_iterations = 4000
n_classes = 2



mode = sys.argv[1]

# Load data
if os.path.isfile("data.h5"):
    # Load Y
    hf = h5py.File('data.h5', 'r')
    y_train = np.array(hf.get('y_train'))
    y_test = np.array(hf.get('y_test')) 
    hf.close()

    # Load X 
    if mode == "one_beat":     
	x_file = h5py.File('one_beat.h5', 'r')
    elif mode == "three_beat":
	x_file = h5py.File('three_beat.h5', 'r')
    else: 
	x_file = h5py.File('two_beat.h5', 'r')

    X_train = np.array(x_file.get('X_train'))
    X_test = np.array(x_file.get('X_test')) 
    x_file.close()
    pdb.set_trace()
    # Load test patients
    test_hf = h5py.File('test_patients.h5', 'r')
    test_patients = np.array(test_hf.get('test_patients'))
    test_patient_labels = np.array(test_hf.get('test_patient_labels'))
    test_hf.close()
else:
    X_train, y_train, X_test, y_test, test_patients, test_patient_labels = loadECG(train_normal_ids, train_death_ids, test_normal_ids, test_death_ids)

# Pre-process data
X_train = np.swapaxes(X_train, 1, 2)
X_test = np.swapaxes(X_test, 1, 2)
new_order = np.arange(len(X_train))
np.random.shuffle(new_order)
X_train = X_train[new_order]
y_train = y_train[new_order]

new_order = np.arange(len(X_test))
np.random.shuffle(new_order)
X_test = X_test[new_order]
y_test = y_test[new_order]

X_val = X_test[:30000]
y_val = y_test[:30000]

# Load model
input_dim = X_train.shape[1]
m, embedding_m = build_fc_model((input_dim, 1))
m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
m.summary()


print("loaded data")
hrs = []
discrete_hrs = []
auc_vals = []
for i in range(400):
    # Using a neural network
    m.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=10, verbose=True, batch_size=2000)
 	
 	test_patients = np.resize(X_test, (627, 1000, input_dim, 1))
    scores = risk_scores(m, test_patients)
    discrete_scores = [1 if x > np.percentile(scores, 75) else 0 for x in scores]
    
    auc_val = evaluate_AUC(scores, test_patient_labels)
    hr = evaluate_HR(scores, test_patients, test_patient_labels)
    discrete_hr = evaluate_HR(discrete_scores, test_patients, test_patient_labels)
    
    auc_vals.append(auc_val)
    hrs.append(hr)
    discrete_hrs.append(hr)
  
    pdb.set_trace()

print("lol")
