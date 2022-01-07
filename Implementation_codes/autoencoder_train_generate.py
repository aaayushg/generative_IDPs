#! /usr/bin/python3.6

import sys, os
import shutil
from copy import deepcopy
import _pickle as cPickle

import scipy.spatial.distance as S

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential, load_model

import biobox as bb
import random as rn

np.random.seed(123)
rn.seed(123)
tf.random.set_seed(1)

### Parser reading input parameters ###
class Parser:
    
    parameters={}

    def __init__(self):

        self.add('dimensions','encoding_dim','int', -1)
        self.add('out_label','outlabel','str', "protein")

        self.add('trajectory','fname','str', "")
        self.add('target_struct','fname2', 'str', "")

        self.add('test_size','test_size','int', 100) 
        self.add('epochs','epochs','int', 1000)
        self.add('batch_size','batch_size','int', 200) 
        self.add('out_folder','outfolder','str', "result")
        self.add('optimizer','optimizer','str', "adam")

        self.add('decoder_file', 'decoderfile', 'str', "")
        self.add('encoder_file', 'encoderfile', 'str', "")

        self.set_default_values()


    #insert a new keyword entry in parameters dictionary Xx enter all the tuples above in the self.parameter xX
    def add(self,key,variable,vartype,default):
        self.parameters[key]=[variable,vartype,default]


    #set default values for all defined keywords Xx Parsing the self paramater and creating a self of all the values xX
    def set_default_values(self):
        for k,v in self.parameters.items():
            exec('self.%s=v[2]'%v[0]) # -> this is where the self.style comes from


    #parse input file and replace default values
    def parse(self,infile):

        f = open(infile, 'r+')
        line = f.readline()
        while line:
            w = line.split()

            if len(w) > 0 and str(w[0][0])!='#':

            #val=[variable_name,variable_type,default_value] Xx same as the one above xX
                try:
                    val=self.parameters[w[0]] # -> get the default parameter of the value?
                except KeyError as e:
                    print("> Unrecognised keyword %s"%w[0])
                    sys.exit(1)

                #if type is string
                if val[1].split()[0]=='str': # _> val[1] = 'str' or 'int'
                    exec('self.%s=%s("%s")'%(val[0],val[1],w[1])) # Xx here you are replacing the default by input paramater
                #if type is int or float
                elif val[1].split()[0]=='int' or val[1].split()[0]=='float':
                    exec('self.%s=%s(%s)'%(val[0],val[1],w[1]))

                # in case the variable_name is a monomer or trajectory having multiple files (case of Hetero-multimer assembly)
                elif val[1].split()[0]=='array' and (val[1].split()[1]=='int' or val[1].split()[1]=='float' or val[1].split()[1]=='str')\
                    and (val[0] == "monomer_file_name" or val[0] == "topology" or val[0] == "trajectory") :
                    exec("test = self.%s" % (val[0]))
                    if (test == "NA") :
                        exec("self.%s = []" % (val[0]))
                        exec('self.%s += [np.array(%s).astype(%s)]' % (val[0],w[1:len(w)],val[1].split()[1]))

                    else:
                        exec('self.%s += [np.array(%s).astype(%s)]' % (val[0],w[1:len(w)],val[1].split()[1]))


                #if type is an array of int, float, or str
                elif val[1].split()[0]=='array' and (val[1].split()[1]=='int' or val[1].split()[1]=='float' or val[1].split()[1]=='str'):
                    exec('self.%s=np.array(%s).astype(%s)'%(val[0],w[1:len(w)],val[1].split()[1]))

                else:
                    print("> Unrecognised type for keyword %s: %s"%(w[0],val[1]))
                    sys.exit(1)

            line = f.readline()

        f.close()

##############
# PARAMETERS #
##############

infile = sys.argv[1]
if not os.path.isfile(infile):
    print("> input file %s not found"%infile)

params = Parser()
params.parse(infile)

encoding_dim = params.encoding_dim #3 # number of floating points in latent vector (or number of eigenvectors)
outlabel = params.outlabel #"1E6J" # label for output files

# input data
fname = params.fname
fname2 = params.fname2 

# training parameters
test_size = params.test_size #100 # number of items in test set
epochs = params.epochs #2000 # number of training epochs for neural network
batch_size = params.batch_size #200 #batch size
outfolder = params.outfolder #"result_%s/%s_neurons_3"%(outlabel, encoding_dim) # folder where to dump all results
optimizer = params.optimizer

decoderfile = params.decoderfile #"%s/decoder_%s_neurons.h5"%(outfolder, encoding_dim)
encoderfile = params.encoderfile #"%s/encoder_%s_neurons.h5"%(outfolder, encoding_dim)

#############################
# LOAD AND ALIGN SIMULATION #
#############################

#make output directory, if unexistent, and copy there a backup of input file
if not os.path.isdir(outfolder):
    os.mkdir(outfolder)

shutil.copy(infile, outfolder)

print("\n> writing results in folder ", outfolder)

print("\n> loading molecule ", fname)
M = bb.Molecule()
M.import_pdb(fname)

if False:
    #align all structures, and center to origin
    idx1 = M.atomselect("*", "*", "CA", get_index=True)[1]
    M.rmsd_one_vs_all(0, points_index=idx1, align=True)

    for pos in range(len(M.coordinates)):
       M.set_current(pos)
       M.center_to_origin()
       print(">> aligning %s"%pos)
    M.set_current(0)


###########################################
# SELECT ATOMS FOR TRAINING AND TEST SETS #
###########################################

# atom selection for reconstruction
idx = M.atomselect("*", "*", ["CA", "CB", "C", "N", "CG" , "CD", "NE2", "CH3", "O", "OE1"], get_index=True)[1]

crds = M.coordinates[:, idx]
print("> molecule has %s conformations"%M.coordinates.shape[0])
print("> selected %s atoms"%len(idx))

#randomly separate simulation in training and test set
print("> extracting %s random items for test set"% test_size)
#indices = np.random.permutation(len(crds))
indicestrain = np.arange(0,20000,1)

# save structures of test set for later comparison
indicestest = np.arange(20000,95000,1)
test_structs = M.get_subset(idxs=idx, conformations=indicestest)
test_structs.write_pdb("%s/%s_test_set.pdb"%(outfolder, outlabel))
train_structs = M.get_subset(idxs=idx, conformations=indicestrain) 
#train_structs.write_pdb("%s/%s_train_set.pdb"%(outfolder, outlabel))

###################################
# PREPARE DATA FOR NEURAL NETWORK #
###################################

x_train = deepcopy(crds[indicestrain])
x_train = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))

x_test = deepcopy(crds[indicestest])
x_test = x_test.reshape(x_test.shape[0], np.prod(x_test.shape[1:]))

print("\n> input data shape: ", crds.shape)
print("> training dataset size: %s"%x_train.shape[0])
print("> testing dataset size: %s"%x_test.shape[0])
print("> protein d.o.f.: %s"%x_train.shape[1])
print("> compression factor:", x_train.shape[1]/encoding_dim)

bounds = 1.0

#normalize data
scaling = np.max(x_train)*bounds
x_train /= scaling
scaling2 =  np.max(x_test)*bounds
x_test /= scaling2


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
decoder.save(decoderfile)
encoder.save(encoderfile)

################
# RUN TEST SET #
################

# encode and decode all the training set, so now we can compare the two
encoded_test_prots = encoder.predict(x_test)
decoded_test_prots = decoder.predict(encoded_test_prots)

#writing decoded structures
print("> writing all encoded-decoded structures...")
decoded_reshaped = decoded_test_prots.reshape(decoded_test_prots.shape[0], int(decoded_test_prots.shape[1]/3), 3)
M3 = M.get_subset(idxs=idx, conformations=[0])
M3.coordinates = decoded_reshaped*scaling2
M3.write_pdb("%s/%s_decoded_%s_neurons.pdb"%(outfolder, outlabel, encoding_dim))

encoded_train_prots = encoder.predict(x_train)

mean_a= np.mean(encoded_train_prots, axis=0)
cov=np.cov(encoded_train_prots.T)

cov_prots = np.random.multivariate_normal(mean_a,cov,size=75000)
decoded_cov_prots = decoder.predict(cov_prots)

decoded_reshaped2 = decoded_cov_prots.reshape(decoded_cov_prots.shape[0], int(decoded_cov_prots.shape[1]/3), 3)
M5 = M.get_subset(idxs=idx, conformations=[0])
M5.coordinates = decoded_reshaped2*scaling2
M5.write_pdb("%s/%s_cov_%s_neurons.pdb"%(outfolder, outlabel, encoding_dim))

#save all encodings
np.savetxt("%s/%s_encoded_training_set.dat"%(outfolder, outlabel), encoded_train_prots)
np.savetxt("%s/%s_encoded_test_set.dat"%(outfolder, outlabel), encoded_test_prots)
np.savetxt("%s/%s_cov_set.dat"%(outfolder, outlabel), cov_prots)

