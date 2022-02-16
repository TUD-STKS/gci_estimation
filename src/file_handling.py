"""
File handling utilities required to reproduce the results in the paper
'Glottal Closure Instant Detection using Echo State Networks'.
"""
# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3-Clause

import os
import numpy as np
import csv


def get_file_list(directory):
    directory = os.path.abspath(directory)
    return [os.path.join(directory, f) for f in os.listdir(directory)]


def train_test_split(file_list):
    training_files = file_list[:8]
    test_files = file_list[8:12]
    return training_files, test_files


def write_annotation_file(path, intervals, annotations=None):
    if annotations is not None and len(annotations) != len(intervals):
        raise ValueError('len(annotations) != len(intervals)')

    with open(path, 'w') as output_file:
        writer = csv.writer(output_file, delimiter=',')

        if annotations is None:
            for t_int in intervals:
                writer.writerow(['%0.3f' % t_int[0], '%0.3f' % t_int[1]])
        else:
            for t_int, lab in zip(intervals, annotations):
                writer.writerow(['%0.3f' % t_int[0], '%0.3f' % t_int[1], lab])


def annot_to_time_series(annot, intervals, hop_len):
    times = np.arange(start=0, stop=intervals[-1, 1], step=0.01)
    annotations = np.zeros(shape=times.shape)
    for count, interval in enumerate(intervals):
        annotations[np.multiply(times >= interval[0], times < interval[1])] =\
            annot[count]
    return times, annotations
