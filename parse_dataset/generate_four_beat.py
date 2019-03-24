import h5py
import numpy as np
import pdb

dataset_dir = "./datasets/splits/"
splits = ["0", "1", "2", "3", "4"]
n_beats = 1000
for split in splits:
	print("Split: ", split)
	split_dir = dataset_dir + "/split_" + split 
	
	train_file = h5py.File(split_dir + "/train_three.h5")
	last_beat_file = h5py.File(split_dir + "/train_one.h5")
	first_three_beats = train_file['adjacent_beats'][:,:n_beats,:]
	last_beat = last_beat_file['adjacent_beats'][:,2:n_beats+2,:128]
	adj_beats = np.concatenate([first_three_beats,last_beat], axis=2)
	mi_labels = train_file['mi_labels'][:]
	cvd_labels = train_file['cvd_labels'][:]

	train_new = h5py.File(split_dir + "/train_four.h5")
	train_new.create_dataset("adjacent_beats", data=adj_beats)
	train_new.create_dataset("mi_labels", data=mi_labels)
	train_new.create_dataset("cvd_labels", data=cvd_labels)
	train_file.close()
	train_new.close()	
	last_beat_file.close()
	
	test_file = h5py.File(split_dir + "/test_three.h5")
	last_beat_file = h5py.File(split_dir + "/test_one.h5")
	first_three_beats = test_file['adjacent_beats'][:,:n_beats,:]
	last_beat = last_beat_file['adjacent_beats'][:,2:n_beats+2,:128]
	adj_beats = np.concatenate([first_three_beats,last_beat], axis=2)
	mi_labels = test_file['mi_labels'][:]
	cvd_labels = test_file['cvd_labels'][:]

	test_new = h5py.File(split_dir + "/test_four.h5")
	test_new.create_dataset("adjacent_beats", data=adj_beats)
	test_new.create_dataset("mi_labels", data=mi_labels)
	test_new.create_dataset("cvd_labels", data=cvd_labels)
	test_file.close()
	test_new.close()
	last_beat_file.close()
