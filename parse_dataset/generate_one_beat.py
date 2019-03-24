import h5py 
import numpy as np
import pdb
import os 
dataset_dir = "./datasets/splits/"
splits = ["0", "1", "2", "3", "4"]

for split in splits:
	split_dir = dataset_dir + "split_" + split
	train_file = h5py.File(split_dir + "/train.h5", "r")
	
	# create new train file
	if os.path.isfile(split_dir + "/train_one.h5"):
		continue
	adj_beats = train_file['adjacent_beats'][:,:,:128]
	cvd_labels = train_file['cvd_labels'][:]
	mi_labels = train_file['mi_labels'][:]

	train_one_beat = h5py.File(split_dir + "/train_one.h5", "w")
	train_one_beat.create_dataset('adjacent_beats', data=adj_beats)
	train_one_beat.create_dataset('cvd_labels', data=cvd_labels)
	train_one_beat.create_dataset('mi_labels', data=mi_labels)
	train_file.close()
	train_one_beat.close()
	
	# create new test file
	if os.path.isfile(split_dir + "/test_one.h5"):
		continue
	
	test_file = h5py.File(split_dir + "/test.h5", "r")
	adj_beats = test_file['adjacent_beats'][:,:,:128]
	cvd_labels = test_file['cvd_labels'][:]
	mi_labels = test_file['mi_labels'][:]
	
	test_one_beat = h5py.File(split_dir + "/test_one.h5", "w")
	test_one_beat.create_dataset('adjacent_beats', data=adj_beats)
	test_one_beat.create_dataset('cvd_labels', data=cvd_labels)
	test_one_beat.create_dataset('mi_labels', data=mi_labels)
	test_file.close()
	test_one_beat.close()

