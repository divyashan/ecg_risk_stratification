import os
import pdb
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_DIR = "../jiffy_experiments/adjacent_beats/"

def get_dataset_path(pid):
    return DATASET_DIR + "patient_" + str(pid) + ".csv"

def get_f_path(pid, folder):
    f_path_poss =  "./datasets/adjacent_beats/" + folder + "/patient_" + str(int(pid)) + ".0.csv"
    f_path = None
    if os.path.isfile(f_path_poss):
        f_path = f_path_poss
    return f_path

def get_patient_data(patient_ids, mode="normal"):
    patient_list = []
    for pid in patient_ids:
        f_path = get_f_path(pid, mode)
        if not f_path:
            print "Couldn't find: ", pid
            continue
        pid_mat = pd.read_csv(f_path, delim_whitespace=True).values
        if len(pid_mat) < 1000:
            continue
        patient_list.append(pid_mat[:1000])
    return patient_list

def create_three_beat_arr(two_beat_arr):
    new_col = np.concatenate([two_beat_arr[1:,:128,:], two_beat_arr[0:1,:128,:]])
    three_beat = np.concatenate([two_beat_arr, new_col], axis=1)
    three_beat = np.delete(three_beat, list(range(999, three_beat.shape[0], 1000)), axis=0)
    return three_beat

def loadECG(train_normal, train_death, test_normal, test_death):
    print("Loading data...")
    train_normal_patients = train_normal
    train_death_patients = train_death
    test_normal_patients = test_normal
    test_death_patients = test_death


    train_death_list = get_patient_data(train_death_patients, "death")
    train_normal_list = get_patient_data(train_normal_patients, "normal")
    test_death_list = get_patient_data(test_death_patients, "death")
    test_normal_list = get_patient_data(test_normal_patients, "normal")

    train_death_list = np.concatenate(train_death_list)
    train_normal_list = np.concatenate(train_normal_list)
    test_death = np.concatenate(test_death_list)
    test_normal = np.concatenate(test_normal_list)

    test_x_list = test_death_list + test_normal_list
    test_y_list = [1 for x in range(len(test_death_list))] + [0 for x in range(len(test_normal_list))]

    train = np.concatenate([train_death_list, train_normal_list], axis=0)
    test = np.concatenate([test_death, test_normal], axis=0)
    
    X_train, y_train = train[:,2:], train[:,1]
    X_test, y_test = test[:,2:], test[:,1]
    
    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)
    
    """
    dbfile = h5py.File("data.h5", "w")
    dbfile.create_dataset("y_train", data=y_train)
    dbfile.create_dataset("y_test", data=y_test)
    dbfile.close()

    test_file = h5py.File("test_patients.h5", "w")
    test_file.create_dataset("test_patients", data=test_x_list)
    test_file.create_dataset("test_patient_labels", data=test_y_list)
    test_file.close()

    two_beat = h5py.File("two_beat.h5", "w")
    two_beat.create_dataset("X_train", data=X_train)
    two_beat.create_dataset("X_test", data=X_test)
    two_beat.close()

    one_beat = h5py.File("one_beat.h5", "w")
    one_beat.create_dataset("X_train", data=X_train[:,:128,:])
    one_beat.create_dataset("X_test", data=X_test[:,:128,:])
    one_beat.close()
    """

    three_beat = h5py.File("three_beat.h5", "w")
    three_beat.create_dataset("X_train", data=create_three_beat_arr(X_train))
    three_beat.create_dataset("X_test", data=create_three_beat_arr(X_test))
    three_beat.close()

    return X_train, y_train, X_test, y_test, test_x_list, test_y_list
