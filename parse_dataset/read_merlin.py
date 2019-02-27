import os
import sys
import pdb
import h5py

import numpy as np
from scipy.io import loadmat
sys.path.insert(0, '../')

# Read patient_outcomes.mat
# Produce first hour and first 24 hour embeddings
DATA_PATH = "/Volumes/My Book/merlin_final/"
META_PATH = "./datasets/all_ids"
WRITE_PATH = "./datasets/patient_data_"
MAT_SUFFIX = "1_thk.mat"

CV_DEATH_COL = 1
CV_DAYS_COL = 4
SIGNAL_LENGTH = 460800
BEAT_LENGTH = 128
BLOCK_SIZE = 5


def get_adjacent_beats(pid):
	# Returns matrix of adjacent_beats 

    f_path = DATA_PATH + str(int(pid)) + "/1filt.mat"
    if os.path.isfile(f_path):
        patient_mat = sio.loadmat(f_path)
    else:
        return np.array([None])

    # Collect ranges of good beats
    beats = patient_mat['good_bts'].flatten()
    beats = [beat_idx for beat_idx in beats if beat_idx < beats[0]+460800-64]

    if not beats:
        print("No good beats for pid: ", pid)
        return np.array([None])
    # Stop signal after first hour of 'good' beats
    signal = patient_mat['signal'].flatten()
    try:
        signal = signal[:beats[0]+460800]
    except:
        pdb.set_trace()
    all_adjacent_beats = []
    for i in range(len(beats)-1):
        beat_peak_1 = beats[i]
        beat_peak_2 = beats[i + 1]
        current_signal = signal[beat_peak_1-(BEAT_LENGTH/2):beat_peak_1+(BEAT_LENGTH/2)]
        next_signal = signal[beat_peak_2-(BEAT_LENGTH/2):beat_peak_2+(BEAT_LENGTH/2)]

        patient_signal = np.concatenate((current_signal, next_signal))
        if len(patient_signal) == 0:
            continue
        all_adjacent_beats.append(patient_signal)
    return np.array(all_adjacent_beats)

def get_pid_fname(pid):
    fname = "./datasets/patient_h5py/" + str(int(pid)) + ".h5"
    return fname

def get_pid_labels(pids, type="cvd"):
    patient_outcomes = loadmat("./datasets/patient_outcomes.mat")['outcomes'] 
    if type == "cvd":
        days_map = {x[0]: x[4] for x in patient_outcomes}
        event_map = {x[0]: x[7] for x in patient_outcomes}
    else:
        days_map = {x[0]: x[3] for x in patient_outcomes}
        event_map = {x[0]: x[1] for x in patient_outcomes}
    
    for key in days_map:
        if event_map[key] == 0:
            days_map[key] = 0
    return [days_map[pid] for pid in pids]

def process_patient_beats(patient_beats):
    if len(patient_beats) > 3600:
        return patient_beats[:3600]
    return patient_beats

def aggregate_h5py(pids):
    # create array of data from pid data
    n_pid_beats = 3600
    n_pids = len(pids)
    all_pid_data = np.zeros((n_pids, n_pid_beats, 256))
    validated_pids = []
    valid_idx = 0
    for i,pid in enumerate(pids):
        fname = get_pid_fname(pid)
        if not os.path.isfile(fname):
            print("No file for: ", pid)
            validated_pids.append(-1)
            continue 
        print("Processing: ", pid)
        patient_file = h5py.File(fname, "r")
        patient_beats = patient_file.get("adjacent_beats")
        n_rows = min(patient_beats.shape[0], 3600)
        all_pid_data[valid_idx,:patient_beats.shape[0],:patient_beats.shape[1]] = process_patient_beats(patient_beats)
        patient_file.close()
        validated_pids.append(pid)
        valid_idx += 1
    return validated_pids, all_pid_data

def write_adjacent_beats():
    all_pids = np.loadtxt(META_PATH, delimiter=",")

    print("Starting on patient signal processing...")
    for pid in all_pids:
        fname = get_pid_fname(pid)
        if os.path.isfile(fname):
            print("Already created: ", fname)
            continue
        print("Getting adjacent beats for: ", pid)
        beats = get_adjacent_beats(pid)
        if not beats.all() or not beats.any():
            print("error in get_adjacent_beats" + str(int(pid)))
            continue

        patient_file = h5py.File(fname, "w")
        patient_file.create_dataset("adjacent_beats", data=beats)
        patient_file.close()

        print("Processed patient: ", pid)

def write_valid_ids():
    dir_prefix = "datasets/splits/split_"
    n_splits = 5
    valid_train_ids = []
    valid_test_ids = []
    for i in range(n_splits):
        train_fname = dir_prefix + str(int(i)) + "/train.h5"
        test_fname = dir_prefix + str(int(i)) + "/test.h5"

        train_h5py = h5py.File(train_fname, "r")
        valid_train_ids.append(np.array(train_h5py.get("pids")))
        train_h5py.close()

        test_h5py = h5py.File(test_fname, "r")
        valid_test_ids.append(np.array(test_h5py.get("pids")))
        test_h5py.close()
    pdb.set_trace()
    np.savetxt("./datasets/valid_test_ids", valid_test_ids)
    np.savetxt("./datasets/valid_train_ids", valid_train_ids)


def write_splits():
    dir_prefix = "datasets/splits/split_"
    n_splits = 5
    train_pids = np.loadtxt('./datasets/train_ids')
    test_pids = np.loadtxt('./datasets/test_ids')

    valid_train_pids = []
    valid_test_pids = []
    for i in range(n_splits):
        print("Processing Split #", i)
        train_split = train_pids[i]
        test_split = test_pids[i]

        train_fname = dir_prefix + str(int(i)) + "/train.h5"
        if not os.path.isfile(train_fname):
            valid_train_split, train_data = aggregate_h5py(train_split)
            train_h5py = h5py.File(train_fname, "w")
            train_h5py.create_dataset("adjacent_beats", data=train_data)
            train_h5py.create_dataset("pids", data=valid_train_split)
            train_h5py.create_dataset("mi_labels", data=get_pid_labels(valid_train_split, "mi"))
            train_h5py.create_dataset("cvd_labels", data=get_pid_labels(valid_train_split, "cvd"))
            train_h5py.close()
            valid_train_pids.append(valid_train_split)

        test_fname = dir_prefix + str(int(i)) + "/test.h5"
        if not os.path.isfile(test_fname):
            valid_test_split, test_data = aggregate_h5py(test_split)
            test_h5py = h5py.File(test_fname, "w")
            test_h5py.create_dataset("adjacent_beats", data=test_data)
            test_h5py.create_dataset("pids", data=valid_test_split)
            test_h5py.close()
            valid_test_pids.append(valid_test_split)

    np.savetxt("./datasets/valid_train_ids", np.array(valid_train_pids))
    np.savetxt("./datasets/valid_test_ids", np.array(valid_test_pids))



# def write_labels():
#     dir_prefix = "datasets/splits/split_"
#     n_splits = 5
#     train_pids = np.loadtxt('./datasets/valid_train_ids')
#     test_pids = np.loadtxt('./datasets/valid_test_ids')

#     # Here we're saving the days as labels - this makes the threshold for prediction - 60, 90, whatever - a choice made in the experiment
#     patient_outcomes = loadmat("./datasets/patient_outcomes.mat")['outcomes'] 
#     cvd_days_map = {x[0]: x[4] for x in patient_outcomes}
#     mi_days_map = {x[0]: x[3] for x in patient_outcomes}
#     cvd_event_map = {x[0]: x[7] for x in patient_outcomes}
#     mi_event_map = {x[0]: x[1] for x in patient_outcomes}

#     for key in cvd_days_map:
#         if cvd_event_map[key] == 0:
#             cvd_days_map[key] = 0
#         if mi_event_map[key] == 0:
#             mi_days_map[key] = 0

#     for i in range(n_splits):
#         train_split = train_pids[i]
#         test_split = test_pids[i]
#         cvd_train_fname = dir_prefix + str(int(i)) + "/cvd_train_labels"
#         cvd_test_fname = dir_prefix + str(int(i)) + "/cvd_test_labels"
#         mi_train_fname = dir_prefix + str(int(i)) + "/mi_train_labels"
#         mi_test_fname = dir_prefix + str(int(i)) + "/mi_test_labels"

#         np.savetxt(cvd_train_fname, get_pid_labels(train_split, cvd_days_map))
#         np.savetxt(mi_train_fname, get_pid_labels(train_split, mi_days_map))
#         np.savetxt(cvd_test_fname, get_pid_labels(test_split, cvd_days_map))
#         np.savetxt(mi_test_fname, get_pid_labels(test_split, mi_days_map))
        

def add_labels_to_hd5files():
    dir_prefix = "datasets/splits/split_"
    parts = ["train", "test"]
    n_splits = 5
    for i in range(n_splits):
        for part in parts:
            split_dir = dir_prefix + str(int(i)) 
            data_fname = split_dir + "/" + part + ".h5"
            data_f = h5py.File(data_fname, "w")
            cvd_labels = np.loadtxt(split_dir + "/" + "cvd_" + part + "_labels")
            mi_labels = np.loadtxt(split_dir + "/" + "mi_" + part + "_labels")

            data_f.create_dataset("cvd_labels", data=cvd_labels)
            data_f.create_dataset("mi_labels", data=mi_labels)
            data_f.close()



# write_adjacent_beats()
write_splits()
# write_valid_ids()
# write_labels()
add_labels_to_hd5files()
print("lol")
