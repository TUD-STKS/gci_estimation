"""
Preprocessing utilities required to reproduce the results in the paper
'Glottal Closure Instant Detection using Echo State Networks'.
"""
# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3-Clause

import numpy as np
import librosa
from scipy.signal import find_peaks
import csv


def peak_picking(y, thr):
    return librosa.util.peak_pick(y, pre_max=1, post_max=1, pre_avg=1,
                                  post_avg=1, delta=thr, wait=10)


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


def binarize_signal(y, thr=0.04):
    y_diff = np.maximum(np.diff(y, prepend=0), thr)
    peaks, _ = find_peaks(y_diff)
    y_bin = np.zeros_like(y_diff, dtype=int)
    y_bin[peaks] = 1
    return y_bin


def extract_features(training_files: str, test_files: str, sr: float = 4000.,
                     frame_length: int = 81, target_widening: bool = True):
    X_train = np.empty(shape=(len(training_files)), dtype=object)
    y_train = np.empty(shape=(len(training_files)), dtype=object)
    X_test = np.empty(shape=(len(test_files)), dtype=object)
    y_test = np.empty(shape=(len(test_files)), dtype=object)
    for k, f in enumerate(training_files):
        s, sr = librosa.load(f, sr=sr, mono=False)
        # s[0, :] = librosa.util.normalize(s[0, :])
        X_train[k] = librosa.util.frame(
            s[0, :], frame_length=frame_length, hop_length=1).T
        y_train[k] = librosa.util.frame(
            binarize_signal(s[1, :], 0.04), frame_length=frame_length,
            hop_length=1).T
        if target_widening:
            y_train[k] = np.convolve(y_train[k][:, int(frame_length / 2)],
                                     [0.5, 1.0, 0.5], 'same').reshape(-1, 1)
        else:
            y_train[k] = y_train[k][:, int(frame_length / 2)].reshape(-1, 1)
    for k, f in enumerate(test_files):
        s, sr = librosa.load(f, sr=sr, mono=False)
        # s[0, :] = librosa.util.normalize(s[0, :])
        X_test[k] = librosa.util.frame(
            s[0, :], frame_length=frame_length, hop_length=1).T
        y_test[k] = librosa.util.frame(
            binarize_signal(s[1, :], 0.04), frame_length=frame_length,
            hop_length=1).T
        y_test[k] = y_test[k][:, int(frame_length / 2)].reshape(-1, 1)

    return X_train, X_test, y_train, y_test
