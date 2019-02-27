import os
import sys
import h5py
import pdb
import numpy as np
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
import tftables
restrict_GPU_keras("1")

mode = sys.argv[1]
m_type = sys.argv[2]

y_mode = "cvd"
splits = ["0", "1", "2", "3", "4"]
split_num = "0"
split_dir = "./datasets/splits/split_" + split_num
# Load Y
# hf = h5py.File('datasets/data.h5', 'r')
# y_train = np.array(hf.get('y_train'))
# y_test = np.array(hf.get('y_test')) 
# hf.close()

# Load Y
y_train = np.loadtxt(split_dir + "/" + y_mode + "_train_labels")
y_test = np.loadtxt(split_dir + "/" + y_mode + "_test_labels")


# Load X 
# if mode == "one_beat":     
#     x_file = h5py.File('datasets/one_beat.h5', 'r')
#     all_test = h5py.File('datasets/all_test_one.h5', 'r')
#     n_beats = 1000
#     instance_length = 128
# elif mode == "three_beat":
#     x_file = h5py.File('datasets/three_beat.h5', 'r')
#     all_test = h5py.File('datasets/all_test_three.h5', 'r')
#     n_beats = 998
#     instance_length = 384
# elif mode == 'four_beat':
#     x_file = h5py.File('datasets/four_beat.h5', 'r')
#     all_test = h5py.File('datasets/all_test_four.h5', 'r')
#     n_beats = 997
#     instance_length = 512
# else: 
#     x_file = h5py.File('datasets/two_beat.h5', 'r')
#     all_test = h5py.File('datasets/all_test_two.h5', 'r')
#     n_beats = 999 
#     instance_length = 256


# Load X

reader = tftables.open_file(filename=split_dir + "/train.h5", batch_size=10)

x_train_file = h5py.File(split_dir + "/train.h5")
x_test_file = h5py.File(split_dir + "/test.h5")
X_train = np.array(x_train_file.get('adjacent_beats'))
X_test = np.array(x_test_fileß.get('adjacent_beats'))
x_train_file.close()
x_test_file.close()

pdb.set_trace()


# Load X_train, y_train, X_test and y_test
# pdb.set_trace()
# X_train = np.array(x_file.get('X_train'))
# X_test = np.array(x_file.get('X_test')) 
# x_file.close()

all_test_patients = np.array(all_test.get('test_patients'))
all_test_patient_labels = np.array(all_test.get('test_patient_labels'))
all_test_pids = np.loadtxt('all_test_pids')

# # Create balanced 80/20 split
# train_add = all_test_patients[:4000].reshape(4000*n_beats, 1, instance_length) 
# train_label_add = np.array([[g]*n_beats for g in all_test_patient_labels[:4000]])
# X_train = np.concatenate([X_train,train_add])
# y_train = np.concatenate([y_train, train_label_add.reshape(4000*n_beats)])

# all_test_patients = all_test_patients[4000:]
# all_test_patient_labels = all_test_patient_labels[4000:]
# all_test_pids = all_test_pids[4000:]

# # Load test patient metadata
# test_hf = h5py.File('datasets/test_pids.h5', 'r')
# test_pids = np.array(test_hf.get('pids'))
# test_patient_labels = np.array(test_hf.get('patient_labels'))
# test_hf.close()


# Model specific variables
if m_type == 'LR':
    n_iter = 1
    m = LogisticRegression()

    X_train = np.squeeze(X_train, 2)
    test_patients = np.resize(X_test, (627, 1000, input_dim))
    train_m = lambda : m.fit(X_train, y_train)
    score_m = lambda test_patients: risk_scores(m, test_patients)

else:
    n_iter = 40
    m, embedding_m = build_fc_model((input_dim, 1))
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    m.summary()

    test_patients = np.resize(X_test, (627, 1000, input_dim, 1))
    class_weight = {0: 1.,
                    1: 30.}
    train_m = lambda : m.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=10, verbose=True, batch_size=10000, class_weight=class_weight)
    score_m = lambda test_patients: risk_scores(m, test_patients)


hrs = []
discrete_hrs = []
auc_vals = []
for i in range(n_iter):
    train_m()

    # Calculate risk scores for all test patients
    scores = score_m(test_patients)
    # Subsample test set to achieve desired incidence rate
    scores = scores[:600]
    test_patient_labels = test_patient_labels[:600]
    test_pids = test_pids[:600]
    
    discrete_scores = [1 if x >= np.percentile(scores, 75) else 0 for x in scores]
    """ 
    try: 
        auc_val = evaluate_AUC(scores, test_patient_labels)
        hr = evaluate_HR(survival_dict, scores, test_pids , test_patient_labels, "continuous")
        discrete_hr = evaluate_HR(survival_dict, discrete_scores, test_pids, test_patient_labels, "discrete")
    except:
        hr = 0
        auc = 0
        discrete_hr = 0
    
    n_high_risk_death = np.sum(test_patient_labels[np.argsort(scores)][-150:])

    auc_vals.append(auc_val)
    hrs.append(hr)
    discrete_hrs.append(discrete_hr)

    print(n_high_risk_death) 
    print(auc_val, hr, discrete_hr)
    """
    n_high_risk_death = 10
    discrete_hr = [10]
    if n_high_risk_death > 9 or discrete_hr[0] > 9:
        all_scores = score_m(all_test_patients)
        cutoff = np.percentile(all_scores, 75)
        all_discrete_scores = [1 if x >= cutoff else 0 for x in all_scores]
        
        all_auc = evaluate_AUC(all_scores, all_test_patient_labels)
        all_hr = evaluate_HR(survival_dict, scores, all_test_pids, all_test_patient_labels, "continous")
        all_discrete_hr = evaluate_HR(survival_dict, all_discrete_scores, all_test_pids, all_test_patient_labels, "discrete")
        print("ALL patient scores: ")
        print(all_auc,all_hr, all_discrete_hr)
        pdb.set_trace() 
pdb.set_trace()
