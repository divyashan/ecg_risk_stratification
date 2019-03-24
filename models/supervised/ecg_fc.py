from keras.models import Model
from keras.layers import Input, Flatten, Dense, Reshape, Lambda, Add, Concatenate, concatenate
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv1D, MaxPooling1D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.initializers import glorot_normal
from keras.regularizers import l2
import keras.backend as K
import sys

sys.path.insert(0, '../')
import numpy as np
import pdb
import os

def build_fc_model( img_shape, num_fc_0=2, dtw_init=False):
		initializer = glorot_normal()
		x0 = Input( img_shape, name='Input')
		raw_x = x0
		flattened_raw_x = Flatten()(raw_x)
		fc1 = Dense( num_fc_0, activation='relu', name='fc_1', kernel_initializer=initializer, kernel_regularizer=l2(.001) )(flattened_raw_x)

		#fc1 = Dense( num_fc_1, activation='relu', name='dense_encoding', kernel_initializer=initializer,
		#			 kernel_regularizer=l2(.001) )(fc0)
		y = Dense(1, name='softmax', activation='sigmoid')(fc1)
		embedding_model = Model(inputs = x0, outputs = fc1)
		#embedding_model = Model(inputs = x0, outputs=fc0)
		model = Model( inputs = x0, outputs = y )
		if dtw_init:
			input_dim = 256
			half_dims = (int(input_dim/2), num_fc_0)
			init_weights = np.concatenate([np.ones(half_dims), -1*np.ones(half_dims)])
			rand_bias = model.layers[2].get_weights()[1]
			model.layers[2].set_weights([init_weights, rand_bias])	
		return model, embedding_model
