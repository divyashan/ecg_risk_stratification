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
from sklearn.linear_model import LogisticRegression

from scipy.io import loadmat

sys.path.insert(0, '../')
from ecg_AAAI.parse_dataset.readECG import loadECG
from ecg_AAAI.models.ecg_utils import get_all_adjacent_beats
from ecg_AAAI.models.supervised.ecg_fi_model_keras import build_fi_model 
from ecg_AAAI.models.supervised.ecg_fc import build_fc_model
from ecg_AAAI.models.gpu_utils import restrict_GPU_keras
from ecg_AAAI.models.supervised.eval import evaluate_AUC, evaluate_HR, risk_scores
restrict_GPU_keras("0")

mode = sys.argv[1]
m_type = sys.argv[2]

# Load Y
hf = h5py.File('datasets/data.h5', 'r')
y_train = np.array(hf.get('y_train'))
y_test = np.array(hf.get('y_test')) 
hf.close()

# Load X 
if mode == "one_beat":     
	x_file = h5py.File('datasets/one_beat.h5', 'r')
elif mode == "three_beat":
	x_file = h5py.File('datasets/three_beat.h5', 'r')
elif mode == 'four_beat':
	x_file = h5py.File('datasets/four_beat.h5', 'r')
else: 
	x_file = h5py.File('datasets/two_beat.h5', 'r')

X_train = np.array(x_file.get('X_train'))
X_test = np.array(x_file.get('X_test')) 
x_file.close()

# Load test patient metadata
test_hf = h5py.File('datasets/test_pids.h5', 'r')
test_pids = np.array(test_hf.get('pids'))
test_patient_labels = np.array(test_hf.get('patient_labels'))
test_hf.close()

# Pre-process data
input_dim = X_train.shape[2]
X_train = np.swapaxes(X_train, 1, 2)
X_test = np.swapaxes(X_test, 1, 2)

new_order = np.arange(len(X_train))
np.random.shuffle(new_order)
X_train = X_train[new_order]
y_train = y_train[new_order]

new_order = np.arange(len(X_test))
np.random.shuffle(new_order)
X_val = X_test[new_order][:3000]
y_val = y_test[new_order][:3000]

test_patients = np.resize(X_test, (627, 1000, input_dim, 1))
patient_outcomes = loadmat("./datasets/patient_outcomes.mat")['outcomes'] 
print("loaded data")
hrs = []
discrete_hrs = []
auc_vals = []
if m_type == 'LR':
    m = LogisticRegression()
    m.fit(np.squeeze(X_train, 2), y_train)
else:
    m, embedding_m = build_fc_model((input_dim, 1))
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

for i in range(40):
    if m_type == 'LR':     
        # Using logistic regression
        scores = risk_scores(m, [np.squeeze(x, 2) for x in test_patients])
    else:
        # Using a neural network
        m.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=10, verbose=False, batch_size=4000)	
    	scores = risk_scores(m, test_patients)

    discrete_scores = [1 if x > np.percentile(scores, 75) else 0 for x in scores]

    auc_val = evaluate_AUC(scores, test_patient_labels)
    hr = evaluate_HR(patient_outcomes, scores, test_pids , test_patient_labels, "continuous")
    discrete_hr = evaluate_HR(patient_outcomes, discrete_scores, test_pids, test_patient_labels, "discrete")
    
    score_subset = scores[:600]
    labels_subset = test_patient_labels[:600]
    n_high_risk_death = np.sum(test_patient_labels[np.argsort(score_subset)][-157:])
    print(n_high_risk_death) 
    if n_high_risk_death > 9:
        pdb.set_trace()
    auc_vals.append(auc_val)
    hrs.append(hr)
    discrete_hrs.append(discrete_hr)
    print(auc_val, hr, discrete_hr) 
pdb.set_trace()
