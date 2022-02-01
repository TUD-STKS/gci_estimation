"""
File handling utilities required to reproduce the results in the paper
'Glottal Closure Instant Detection using Echo State Networks'.
"""
# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3-Clause

import os


def get_file_list(directory):
    directory = os.path.abspath(directory)
    return [os.path.join(directory, f) for f in os.listdir(directory)]


def train_test_split(file_list):
    training_files = file_list[:8]
    test_files = file_list[8:12]
    return training_files, test_files
