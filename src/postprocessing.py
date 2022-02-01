"""
Postprocessing utilities required to reproduce the results in the paper
'Glottal Closure Instant Detection using Echo State Networks'.
"""
# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3-Clause

import librosa


def peak_picking(y, thr):
    return librosa.util.peak_pick(y, pre_max=1, post_max=1, pre_avg=1,
                                  post_avg=1, delta=thr, wait=10)
