#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt


from scipy.signal import resample
from scipy.spatial.distance import euclidean

from armada18eeg import *


"""
Does what the program is expected to do, assuming the arguments have been parsed.
directory_path: The directory where the script will look for the files to process.
output_file: The filename of the generated output file.
"""
def do_the_job(input_file):
	# for (dirpath, dirnames, filenames) in os.walk(directory_path):
    for i in range(1):
            full_file_path = input_file
            print ('Using file', input_file)
            vectors, header = generate_feature_vectors_from_samples(full_file_path, samples=150, period=1., 
                                                           state=0
                                                           )
            print ('resulting vector shape for the file', vectors.shape)
            try:
                FINAL_MATRIX = np.vstack( [ FINAL_MATRIX, vectors ] )
            except UnboundLocalError:
                FINAL_MATRIX = vectors

    print ('FINAL_MATRIX', FINAL_MATRIX.shape)
	# np.random.shuffle(FINAL_MATRIX)
	# np.savetxt(output_file, FINAL_MATRIX, delimiter=',', header=header)
    return FINAL_MATRIX,header


"""
Main function. The parameters for the script are the following:
[1] directory_path: The directory where the script will look for the files to process.
[2] output_file: The filename of the generated output file.
"""
if __name__ == '__main__':
	# if len(sys.argv) <3:
	# 	print ('arg1: input dir\narg2: output file')
	# 	sys.exit(-1)
	# directory_path = sys.argv[1]
	# output_file = sys.argv[2]
    input_file=''
    do_the_job(input_file)