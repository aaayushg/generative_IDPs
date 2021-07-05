#! /usr/bin/python3.6
test='polyq_test_set.pdb'
new='polyq_new_200_neurons.pdb'

import sys, os
import shutil
from copy import deepcopy
import biobox as bb
import numpy as np
import heapq
np.set_printoptions(threshold=sys.maxsize)

def compare_structs(test_structs, models):

    # build hybrid stucture (test strcts + decoded strcts)
    test = deepcopy(test_structs)
    for i in range(0, models.coordinates.shape[0]):
        test.add_xyz(models.coordinates[i])
    
    t=[]
    for i in range(0,80000):
        t.append(i)

    m = []
    for i in range(80000,159998):
        r = []
        for j in t:
            val = test.rmsd(i, j)
            r.append(val)
        minimum = min(r)
        min_index=r.index(minimum)
        del t[min_index]
        m.append(minimum)
    print(sum(m)/len(m))

    return np.array(m)

test_size=80000

M = bb.Molecule()
M.import_pdb(test)

idx = M.atomselect("*", "*", ["CA", "CB", "C", "N", "CG" , "CD", "NE2", "CH3", "O", "OE1" ], get_index=True)[1]
crds = M.coordinates[:, idx]

indices = np.random.permutation(len(crds))

# save structures of test set for later comparison
indicestest = np.sort(indices[:test_size])
test_structs = M.get_subset(idxs=idx, conformations=indicestest)

M1 = bb.Molecule()
M1.import_pdb(new)
M3 = M1.get_subset(idxs=idx)
rmsd_nn = compare_structs(test_structs, M3)
print("\n> RMSD of rebuilt vs original structures")
print(">> %5.2f pm %5.2f, min: %5.2f, max: %5.2f"%(np.mean(rmsd_nn), np.std(rmsd_nn), np.min(rmsd_nn), np.max(rmsd_nn)))


