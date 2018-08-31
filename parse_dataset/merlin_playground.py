import numpy as np
import scipy.io as sio
import pdb

xx = sio.loadmat("/Volumes/My Book/merlin_final/1/1filt.mat")
range_list = []
current_range = [xx['good_bts'][0], None]
beats = xx['good_bts'].flatten()
pdb.set_trace()
for i in range(len(beats)):

	if i == len(beats)-1:
		current_range[1] = beats[i] + 256
		range_list.append((current_range[0], current_range[1]))
	elif xx['good_bts'][i+1] > beats[i] + 256:
		current_range[1] =  beats[i] + 256
		range_list.append((current_range[0], current_range[1]))
		current_range[0] = beats[i+1]
		current_range[1] = None

pdb.set_trace()
