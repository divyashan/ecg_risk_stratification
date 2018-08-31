import os
import os
import pdb
import numpy as np

death_ids = [1302, 5036, 1136, 1298, 1357, 2398, 5249, 6005, 3270, 4973, 5633,
       2274, 2357, 3350, 6129, 3298, 3617, 3511, 5787, 5616, 3838, 4659,
       4899, 5187, 3797, 4763, 4890, 5415, 6364, 5964, 3605, 3348, 3390, 2718, 1710, 4913, 3525, 1630, 2617, 3232, 3865,
       3277, 2438, 5908, 2628, 2252, 5194, 3418, 2635, 3216, 3541, 4740,
       3645, 6135, 6311, 1763, 3548, 3245, 4844, 1154, 1050, 2547, 2785,
       3875, 1185, 1921, 3570, 1189, 3643, 4000, 2484, 3319, 1175, 1031,
       1378, 1646, 1273, 1197, 2429]

normal_ids =   np.array([800.0, 801.0, 802.0, 803.0, 804.0, 806.0, 807.0, 808.0, 809.0, 81.0, 810.0, 811.0, 812.0, 813.0, 814.0, 815.0, 816.0, 817.0, 818.0, 
                                819.0, 82.0, 820.0, 821.0, 822.0, 824.0, 825.0, 826.0, 827.0, 829.0, 83.0, 832.0, 833.0, 834.0, 835.0, 837.0, 838.0, 839.0, 84.0, 840.0, 841.0, 842.0, 
                                848.0, 85.0, 852.0, 853.0, 854.0, 855.0, 856.0, 857.0, 858.0, 859.0, 86.0, 860.0, 861.0, 862.0, 863.0, 864.0, 865.0, 866.0, 867.0, 868.0, 869.0, 87.0, 870.0, 871.0, 873.0, 
                                874.0, 875.0, 876.0, 878.0, 88.0, 880.0, 881.0, 882.0, 883.0, 884.0, 886.0, 887.0, 888.0, 89.0, 890.0, 891.0, 892.0, 893.0, 894.0, 895.0, 896.0, 897.0, 9.0, 90.0])

def check_file_exists(pid, mode):
	float_file = os.path.isfile("./datasets/adjacent_beats/" + mode + "/patient_" + str(pid) + ".csv")
	int_file = os.path.isfile("./datasets/adjacent_beats/" + mode + "/patient_" + str(int(pid)) + ".csv")


	merlin_file = "/Volumes/My Book/merlin_final/" + str(pid)

	return os.path.isdir(merlin_file)


DNE = []
exists = []
for pid in death_ids:
	if not check_file_exists(pid, "death"):
		DNE.append(pid)
		print pid 
	else:
		exists.append(pid)

pdb.set_trace()
DNE = []
exists = []
for pid in normal_ids:
	if not check_file_exists(pid, "normal"):
		DNE.append(pid)
		print pid 
	else:
		exists.append(pid)
