#! /usr/bin/python3.6

import biobox as bb
import numpy as np
from copy import deepcopy
from keras.models import Model, Sequential, load_model

fname = '/input_data/polyq/all.pdb'

M = bb.Molecule()
M.import_pdb(fname)
batch_size=40

# atom selection for reconstruction
idx = M.atomselect("*", "*", ["CA", "CB", "C", "N", "CG" , "CD", "NE2", "CH3", "O", "OE1"], get_index=True)[1]

crds = M.coordinates[:, idx]

#randomly separate simulation in training and test set
indicestrain = np.arange(0,19000,1)

# save structures of test set for later comparison
indicestest = np.arange(19000,95000,1)
test_structs = M.get_subset(idxs=idx, conformations=indicestest)
train_structs = M.get_subset(idxs=idx, conformations=indicestrain)

###################################
# PREPARE DATA FOR NEURAL NETWORK #
###################################

x_train = deepcopy(crds[indicestrain])
x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
print(x_train.shape)

x_test = deepcopy(crds[indicestest])
x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))
print(x_test.shape)

bounds = 1.0

#normalize data
scaling = np.max(x_train)*bounds
x_train /= scaling
scaling2 =  np.max(x_test)*bounds
x_test /= scaling2

decoder = load_model('decoder.h5')
encoder = load_model('encoder.h5')

encoded_train_prots = encoder.predict(x_train)

mean_a= np.mean(encoded_train_prots, axis=0)
cov=np.cov(np.array(encoded_train_prots).T)

#Sample with criteria - only positive value 
def cov_prots_gen1(n):
    cov_prots=[]
    while len(cov_prots) < n:
            x=np.random.multivariate_normal(mean_a,cov,size=1)
            if np.all(x>=0):
                    print(x)
                    cov_prots.append(x)
            else:
                    continue
    return np.concatenate(cov_prots, axis=0)

#cov_prots=cov_prots_gen2(43500)

#or just sample randomly
cov_prots = np.random.multivariate_normal(mean_a,cov,size=76000)

np.savetxt("cov_positive_set_101500.dat", cov_prots)
decoded_cov_prots = decoder.predict(cov_prots)

decoded_reshaped2 = decoded_cov_prots.reshape(decoded_cov_prots.shape[0], int(decoded_cov_prots.shape[1]/3), 3)
M5 = M.get_subset(idxs=idx, conformations=[0])
M5.coordinates = decoded_reshaped2*scaling2
M5.write_pdb("cov_positive_101500.pdb")
