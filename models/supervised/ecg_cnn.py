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

def build_cnn(img_shape):
	initializer = glorot_normal()
	x0 = Input( img_shape, name='Input')
	pool_size = 4
	x = Conv1D( 2, kernel_size=128, strides=2, activation='relu', padding='same')(x0)
	x = MaxPooling1D(pool_size)(x) 

	x = Conv1D( 2, kernel_size=64, strides=2, activation='relu', padding='same')(x)
	x = MaxPooling1D(pool_size)(x) 
	x = Flatten()(x)

	y = Dense( 1, name='softmax', activation='sigmoid')(x)
	model = Model( inputs = x0, outputs = y )
	return model

