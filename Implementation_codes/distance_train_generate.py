#! /usr/bin/python3.6

import sys, os
import shutil

from sklearn import preprocessing

import numpy as np
import pandas as pd

import tensorflow
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential, load_model

import random as rn

np.random.seed(123)
rn.seed(123)
os.environ['PYTHONHASHSEED']='0'
tensorflow.random.set_seed(1)
os.environ['TF_DETERMINISTIC_OPS']='1'

##############
# PARAMETERS #
##############

infile = pd.read_csv('/home/aayush/Documents/GenIDP/protein_autoencoder/ab40_low/distance/pdbs/pp.dat', delim_whitespace=True)
#infile = infile.drop('#Frame', axis=1)

encoding_dim = 200
outlabel = 'dihed' 

# training parameters
test_size = 98000
epochs = 100 
batch_size = 40
optimizer = 'adam'

decoderfile = 'decoder.h5' 
encoderfile = 'encoder.h5'

###########################################
# SELECT ATOMS FOR TRAINING AND TEST SETS #
###########################################

scaler=preprocessing.MinMaxScaler()

x_train  = infile[1:42000]
np.savetxt('train.dat', x_train)#, fmt='%1.4f')
x_train = scaler.fit_transform(x_train)
#x_train  = ((x_train + 180) /360)
np.savetxt('train_scaled.dat', x_train)

x_test = infile[42000:140000]
np.savetxt('test.dat', x_test)#, fmt='%1.4f')
x_test = scaler.fit_transform(x_test)
#x_test  = ((x_test + 180) /360)
np.savetxt('test_scaled.dat', x_test)

##################################
# BUILD AND TRAIN NEURAL NETWORK #
##################################

print("> building neural network")

# this is our input placeholder
input_prot = Input(shape=(x_train.shape[1],))

encoded = Dense(300, activation='relu')(input_prot)
encoded = Dense(50, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(50, activation='relu')(encoded)
decoded = Dense(300, activation='relu')(decoded)
decoded = Dense(x_train.shape[1], activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(inputs=input_prot, outputs=decoded)
print("> autoencoder model created!")

# this model maps an input to its encoded representation
encoder = Model(inputs=input_prot, outputs=encoded)

autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))


#build multilayer decoder
decoder = Sequential()
decoder.add(Dense(50, input_shape=(encoding_dim,), activation='relu', weights=autoencoder.layers[-3].get_weights()))
decoder.add(Dense(300, activation='relu', weights=autoencoder.layers[-2].get_weights()))
decoder.add(Dense(x_train.shape[1], activation='sigmoid', weights=autoencoder.layers[-1].get_weights()))

#save encoder and decoder for later usage
#decoder.save(decoderfile)
#encoder.save(encoderfile)

################
# RUN TEST SET #
################

# encode and decode all the training set, so now we can compare the two
encoded_test_prots = encoder.predict(x_test)
decoded_test_prots = decoder.predict(encoded_test_prots)

#writing decoded structure
np.savetxt("decoded_test_prots_scaled.dat", decoded_test_prots)
decoded_test_prots = scaler.inverse_transform(decoded_test_prots)
np.savetxt("decoded_test_prots.dat", decoded_test_prots)#, fmt='%1.4f')

encoded_train_prots = encoder.predict(x_train)

mean_a= np.mean(encoded_train_prots, axis=0)
cov=np.cov(encoded_train_prots.T)

cov_prots = np.random.multivariate_normal(mean_a,cov,size=98000)
decoded_cov_prots = decoder.predict(cov_prots)

np.savetxt("decoded_cov_prots_scaled.dat", decoded_cov_prots)
decoded_cov_prots = scaler.inverse_transform(decoded_cov_prots)
np.savetxt("decoded_cov_prots.dat", decoded_cov_prots)#, fmt='%1.4f')

