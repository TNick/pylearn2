#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retreved from 
https://github.com/jordangumm/pylearn2-helpers/blob/master/csv_to_h5.py

@author: jordangumm
"""
import h5py
import numpy as np
import sys
import os.path
 
if len(sys.argv) != 3:
   print "Usage:", sys.argv[0], "<input_file.txt> <output_file.h5>"
   quit()
 
if os.path.exists(sys.argv[2]):
   print "Output file already exists.  Fail."
   quit()
 
d = np.genfromtxt(sys.argv[1], delimiter=',')
 
print d.shape
 
targets=np.zeros((d.shape[0], 2))
for i in range(d.shape[0]):
   if d[i,0] == 1:
      targets[i,0] = 1
   else:
      targets[i,1] = 1
 
f = h5py.File(sys.argv[2], 'w')
f.create_dataset('X', data=d[:,1:])
f.create_dataset('y', data=targets)
