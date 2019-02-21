import os
import sys
import pdb

import numpy as np
import scipy.io as sio
sys.path.insert(0, '../')

# Read patient_outcomes.mat
# Produce first hour and first 24 hour embeddings
DATA_PATH = "/Volumes/My Book/merlin_final/"
META_PATH = "./datasets/patient_labels"
WRITE_PATH = "./datasets/patient_data_"
MAT_SUFFIX = "1_thk.mat"

CV_DEATH_COL = 1
CV_DAYS_COL = 4
SIGNAL_LENGTH = 460800
BEAT_LENGTH = 128
BLOCK_SIZE = 5


def get_patient_signal(patient_id):
    # Given a signal, return the first hour
    f_path = DATA_PATH + str(int(patient_id)) + "/1filt.mat"
    if os.path.isfile(f_path):
        patient_mat = sio.loadmat(f_path)
    else:
        return "DNE"

    # Collect ranges of good beats
    beats = patient_mat['good_bts'].flatten()
    signal = patient_mat['signal'].flatten()
    current_range = [beats[0], None]
    range_list = []
    for i in range(len(beats)):

        if i == len(beats)-1:
            current_range[1] = beats[i] + 256
            range_list.append((current_range[0], current_range[1]))
        elif beats[i+1] > beats[i] + 256:
            current_range[1] = beats[i] + 256
            range_list.append((current_range[0], current_range[1]))
            current_range[0] = beats[i+1]
            current_range[1] = None

    # Use above ranges to remove bad heartbeats from signal
    patient_signal = []
    while len(patient_signal) < SIGNAL_LENGTH:
        idx_range = range_list.pop(0)
        patient_signal.extend(signal[idx_range[0]:idx_range[1]])
    patient_signal = [str(j) for j in patient_signal][:SIGNAL_LENGTH]
    return ','.join(patient_signal)


def process_dataset():
    outcomes = np.loadtxt(META_PATH, delimiter=",")
    cv_death_ids = outcomes[np.where(outcomes[:, CV_DEATH_COL] == 1)[0], 0]
    cv_survive_ids = outcomes[np.where(outcomes[:, CV_DEATH_COL] == 0)[0], 0]

    n_finished = 0

    n_wrong_length = 0
    line_no = 0
    pid_no = 0
    skipped = []

    pid_no = 0
    for block_no in range(int(np.ceil(float(len(cv_survive_ids))/BLOCK_SIZE))):
        outcome_f = open(WRITE_PATH + "survive_" + str(block_no) + ".csv", "w")

        block_ct = 0
        while pid_no < len(cv_survive_ids)-1 and block_ct < BLOCK_SIZE:
            pid_no += 1
            pid = cv_survive_ids[pid_no]
            psignal = get_patient_signal(pid)
            if psignal == "DNE":
                skipped.append(pid)
                continue
            outline = str(pid) + ",0," + psignal + "\n"
            outcome_f.write(outline)


            line_no += 1
            block_ct += 1
            print line_no
        outcome_f.close()

    print "Number of patients with corrupted signals: ", n_wrong_length

def get_adjacent_beats(pid):
	# Returns two matrices adjacent_beats and adjacent_difference
	# adjacent_beats = Nx256 matrix of 
	# adjacent_difference = Nx128 matrix of differences between adjacent beats

    f_path = DATA_PATH + str(int(pid)) + "/1filt.mat"
    if os.path.isfile(f_path):
        patient_mat = sio.loadmat(f_path)
    else:
        return np.array([None])

    # Collect ranges of good beats
    beats = patient_mat['good_bts'].flatten()
    beats = [beat_idx for beat_idx in beats if beat_idx < beats[0]+460800-64]

    # Stop signal after first hour of 'good' beats
    signal = patient_mat['signal'].flatten()
    signal = signal[:beats[0]+460800]
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

def write_adjacent_beats():
    outcomes = np.loadtxt(META_PATH, delimiter=",")
    cv_death_ids = outcomes[np.where(outcomes[:, CV_DEATH_COL] == 1)[0], 0]
    cv_survive_ids = outcomes[np.where(outcomes[:, CV_DEATH_COL] == 0)[0], 0]

    print "Starting on death patient signals..."
    for pid in cv_death_ids:
        beats = get_adjacent_beats(pid)
        pdb.set_trace()
        if not beats.all():
            print(pid)
            continue
        patient_beat_mat = np.zeros((beats.shape[0], 258))
        patient_beat_mat[:,0] = pid
        patient_beat_mat[:,1] = 1
        patient_beat_mat[:,2:] = beats


        np.savetxt("./datasets/adjacent_beats/death/patient_" + str(pid) + ".csv", patient_beat_mat)

        print "Processed patient: ", pid

    print "\nStarting on normal patient signals..."
    for pid in cv_survive_ids:
        beats = get_adjacent_beats(pid)
        if not beats.all():
            continue
        patient_beat_mat = np.zeros((beats.shape[0], 258))
        patient_beat_mat[:,0] = pid
        patient_beat_mat[:,1] = 0
        patient_beat_mat[:,2:] = beats


        np.savetxt("./datasets/adjacent_beats/normal/patient_" + str(pid) + ".csv", patient_beat_mat)

        print "Processed patient: ", pid

write_adjacent_beats()
pdb.set_trace()
print "lol"