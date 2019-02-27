import os
import sys
import pdb
import h5py

import numpy as np
import scipy.io as sio
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
	# Returns two matrices adjacent_beats 
	# adjacent_beats = Nx256 matrix of 
	# TODO: adjacent_difference = Nx128 matrix of differences between adjacent beats

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

def get_pid_labels(pid, pid_map):
    labels = []
    for pid in pids:
        if pid == -1:
            continue
        labels.append(pid_map[pid])
    return labels

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
        all_pid_data[i,:patient_beats.shape[0],:patient_beats.shape[1]] = process_patient_beats(patient_beats)
        patient_file.close()
        validated_pids.append(pid)
    return validated_pids, all_pid_data

def write_adjacent_beats():
    all_pids = np.loadtxt(META_PATH, delimiter=",")

    print "Starting on patient signal processing..."
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
        patient_beat_mat = np.zeros((beats.shape[0], 258))
        patient_beat_mat[:,0] = pid
        patient_beat_mat[:,1] = 1
        patient_beat_mat[:,2:] = beats

        patient_file = h5py.File(fname, "w")
        patient_file.create_dataset("adjacent_beats", data=beats)
        patient_file.close()

        print "Processed patient: ", pid

def write_valid_ids():
    dir_prefix = "datasets/splits/split_"
    n_splits = 5
    for i in range(n_splits):
        train_fname = dir_prefix + str(int(i)) + "/train.h5"
        test_fname = dir_prefix + str(int(i)) + "/test.h5"

        train_h5py = h5py.File(train_fname, "r")
        valid_train_ids.append(train_h5py.get("pids"))
        train_h5py.close()

        test_h5py = h5py.File(test_fname, "r")
        valid_test_ids.append(test_h5py.get("pids"))
        test_h5py.close()

    np.savetxt("./datasets/valid_test_ids", valid_test_ids)
    np.savetxt("./datasets/valid_train_ids", valid_train_ids)


def write_splits():
    dir_prefix = "datasets/splits/split_"
    n_splits = 5
    train_pids = np.loadtxt('./datasets/train_ids')
    test_pids = np.loadtxt('./datasets/test_ids')

    valid_train_pids = []
    valid_test_pids = []
    remaining_splits = [2,4]
    for i in range(n_splits):
        if i not in remaining_splits:
            continue
        print("Processing Split #", i)
        train_split = train_pids[i]
        test_split = test_pids[i]

        train_fname = dir_prefix + str(int(i)) + "/train.h5"
        if not os.path.isfile(train_fname):
            valid_train_split, train_data = aggregate_h5py(train_split)
            train_h5py = h5py.File(train_fname, "w")
            train_h5py.create_dataset("adjacent_beats", data=train_data)
            train_h5py.create_dataset("pids", data=valid_train_split)
            train_h5py.close()
            valid_train_pids.append(valid_train_split)

        # test_fname = dir_prefix + str(int(i)) + "/test.h5"
        # if not os.path.isfile(test_fname):
        #     valid_test_split, test_data = aggregate_h5py(test_split)
        #     test_h5py = h5py.File(test_fname, "w")
        #     test_h5py.create_dataset("adjacent_beats", data=test_data)
        #     test_h5py.create_dataset("pids", data=valid_test_split)
        #     test_h5py.close()
        #     valid_test_pids.append(valid_test_split)

    np.savetxt("./datasets/valid_train_ids", np.array(valid_train_pids))
    np.savetxt("./datasets/valid_test_ids", np.array(valid_test_pids))



def write_labels():
    dir_prefix = "datasets/split_"
    n_splits = 5
    train_pids = np.loadtxt('./datasets/valid_train_ids')
    test_pids = np.loadtxt('./datasets/valid_test_ids')

    # Here we're saving the days as labels - this makes the threshold for prediction - 60, 90, whatever - a choice made in the experiment
    patient_outcomes = loadmat("./datasets/patient_outcomes.mat")['outcomes'] 
    cvd_days_map = {x[0]: x[4] for x in patient_outcomes}
    mi_days_map = {x[0]: x[3] for x in patient_outcomes}

    for i in range(n_splits):
        train_split = train_pids[i]
        test_split = test_pids[i]

        cvd_train_fname = dir_prefix + str(int(n_splits)) + "/cvd_train.txt"
        cvd_test_fname = dir_prefix + str(int(n_splits)) + "/cvd_test.txt"
        mi_train_fname = dir_prefix + str(int(n_splits)) + "/mi_train.txt"
        mi_test_fname = dir_prefix + str(int(n_splits)) + "/mi_test.txt"

        np.savetxt(cvd_train_fname, get_pid_labels(train_split, cvd_days_map))
        np.savetxt(mi_train_fname, get_pid_labels(train_split, mi_days_map))
        np.savetxt(cvd_test_fname, get_pid_labels(test_split, cvd_days_map))
        np.savetxt(mi_test_fname, get_pid_labels(test_split, mi_days_map))
        




# write_adjacent_beats()
write_splits()
# write_valid_ids()
# write_labels()
print "lol"
