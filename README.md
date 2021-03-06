# Artificial Intelligence Guided Conformational Sampling of Intrinsically Disordered Proteins

**Implementation_codes/**
        
         -autoencoder_train_generate.py: Script to train the autoencoder with the MD trajectory (PDB format) as an input. 
                  (Sample input files can be provided on request)

         -run_python.sh: Running the autoencoder_train_generate.py script with customized entries i.e., epochs, batchsize, input file etc.

         -generate_new_IDPs.py: Script for sampling new vectors and decoding full conformation using trained weights of the autoencoder.

Code runs smoothly with the following versions:
- python3.8 -m pip install tensorflow-gpu==2.4.1
- python3.8 -m pip install keras==2.4.3
- Works fine with CUDA 11.0 which nees to be defined in the LD_LIBRARY_PATH 

**Plot_histogram/** : Script to generate histogram plots used in Figure 3.

**RMSD/** : Modified Bosco Ho C script for many-to-many RMSD and other python scripts for comparing many-to-many and one-to-many RMSDs.






