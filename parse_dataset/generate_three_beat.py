import h5py
import numpy as np
import pdb

dataset_dir = "./datasets/splits/"
splits = ["0", "1", "2", "3", "4"]
n_beats = 1001
for split in splits:
	print("Split: ", split)
	split_dir = dataset_dir + "/split_" + split 
	
	train_file = h5py.File(split_dir + "/train.h5")
	first_two_beats = train_file['adjacent_beats'][:,:n_beats,:]
	first_beat = train_file['adjacent_beats'][:,1:n_beats+1,:128]
	adj_beats = np.concatenate([first_two_beats,first_beat], axis=2)
	mi_labels = train_file['mi_labels'][:]
	cvd_labels = train_file['cvd_labels'][:]

	train_two_beat = h5py.File(split_dir + "/train_three.h5")
	train_two_beat.create_dataset("adjacent_beats", data=adj_beats)
	train_two_beat.create_dataset("mi_labels", data=mi_labels)
	train_two_beat.create_dataset("cvd_labels", data=cvd_labels)
	train_file.close()
	train_two_beat.close()	
	
	test_file = h5py.File(split_dir + "/test.h5")
	first_two_beats = test_file['adjacent_beats'][:,:n_beats,:]
	first_beat = test_file['adjacent_beats'][:,1:n_beats+1,:128]
	adj_beats = np.concatenate([first_two_beats,first_beat], axis=2)
	mi_labels = test_file['mi_labels'][:]
	cvd_labels = test_file['cvd_labels'][:]

	test_two_beat = h5py.File(split_dir + "/test_three.h5")
	test_two_beat.create_dataset("adjacent_beats", data=adj_beats)
	test_two_beat.create_dataset("mi_labels", data=mi_labels)
	test_two_beat.create_dataset("cvd_labels", data=cvd_labels)
	test_file.close()
	test_two_beat.close()
	 
