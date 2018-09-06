import numpy as np
import pandas as pd
import sys 
import pdb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "../")
from time_series.models.supervised.ecg_fc import build_fc_model


p1_opts = [.05*i for i in range(2, 20)]
n_instance_opts = [1000]
N_BAGS = 200
# Define classes as drawn from two different multivariate gaussians with some overlap
class_1 = lambda : np.random.multivariate_normal((1, 1), cov=[[1,0],[0,1]])
class_2 = lambda : np.random.multivariate_normal((1, -1), cov=[[1,0],[0,1]])
#class_3 = lambda x: np.random.multivariate_normal((-2, 2))
#class_4 = lambda x: np.random.multivariate_normal((-2, -2))

def gen_bag(p, n):
	instances = []
	for i in range(n):
		class_choice = np.random.rand()
		if class_choice < p:
			instance = class_1()
		else:
			instance = class_2()
		instances.append(instance)
	return instances

def generate_bags(p_1, p_2, n_bags=N_BAGS, n_instances=1000):
	# return n/2 samples of bag 1 and n/2 samples of bag 2
	bags = []
	bag_labels = []
	for i in range(n_bags/2):
		bags.append(gen_bag(p_1, n_instances))
		bags.append(gen_bag(p_2, n_instances))
		bag_labels.append(1)
		bag_labels.append(0)
	return bags, bag_labels


entries = []
for p1 in p1_opts:
	for p2 in np.arange(.1, p1-.05, .05):
		for n_instances in n_instance_opts:
			entry = {'p1': p1, 'p2': p2, 'n_samples': n_instances}
			bags, bag_labels = generate_bags(p1, p2, n_instances=n_instances)
			train_bags, test_bags = bags[:N_BAGS/2], bags[N_BAGS/2:]
			train_bag_labels, test_bag_labels = bag_labels[:N_BAGS/2], bag_labels[N_BAGS/2:]
			train_instance_labels = np.repeat(train_bag_labels, n_instances)
			test_instance_labels = np.repeat(test_bag_labels, n_instances)

			X_train = np.expand_dims(np.array([item for sublist in train_bags for item in sublist]), 2)
			X_test = np.expand_dims(np.array([item for sublist in test_bags for item in sublist]), 2)
			y_train = np.repeat(train_bag_labels, n_instances)
			y_test = np.repeat(test_bag_labels, n_instances)

			# build model
			m, embedding_m = build_fc_model((2, 1))
			m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

			# train model
			m.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=10, verbose=False)
			
			# evaluate model
			y_pred = m.predict(X_test)
			bag_pred = np.mean(np.reshape(y_pred, (-1, n_instances)), axis=1)

			individual_auc = roc_auc_score(y_test, y_pred)
			aggregate_auc = roc_auc_score(test_bag_labels, bag_pred)

			entry['individual_auc'] =  individual_auc
			entry['aggregate_auc'] = aggregate_auc
			print "P1: ", p1, "\tP2:", p2, "\tI-AUC: ", individual_auc, "\tA-AUC: ", aggregate_auc, '\tn_instances: ', n_instances
			entries.append(entry)
entries = pd.DataFrame(entries)
entries.to_csv("lol")
pdb.set_trace()
print("lol")


