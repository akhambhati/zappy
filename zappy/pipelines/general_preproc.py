# -*- coding: utf-8 -*-
"""
Utilities to Preprocess stimulation data.

Author: Ankit N. Khambhati
Last Updated: 2021/06/17
"""


import numpy as np
from ..sigproc import filters


def ieeg_screening_pipeline(signal, fs, fs_ds=512, antialias_freq=250, drift_freq=0.5):
    """
    Apply a preprocessing cleaning procedure.

    Parameters
    ----------
    signal: np.ndarray, shape: [n_samples]
        Signal before cleaning.

    fs: float
        Current sampling frequency.

    antialias_freq: float
        Corner frequency for the low-pass, anti-alias filter.

    drift_freq: flost
        Corner frequency for the high-pass, drift filter.

    Returns
    -------
    signal: np.ndarray, shape: [n_samples]
        Signal after cleaning.

    fs: float
        Sampling frequency after downsampling.
    """

    # Determine target downsample frequency
    fs_ds_adjust, q = filters.resample_factor(fs, fs_ds)
    nyq_fs_ds_adjust = fs_ds_adjust // 2
    assert antialias_freq < nyq_fs_ds_adjust

    # Anti-alias filtering with iterative method
    # Find closest power of 2
    pow2 = int(np.log2(nyq_fs_ds_adjust / antialias_freq))
    if pow2 > 0:
        corner_freq = nyq_fs_ds_adjust
        for ii in range(pow2):
            corner_freq /= 2
            signal = filters.low_pass_filter(signal, fs,
                    corner_freq=corner_freq, stop_tol=10)
    signal = filters.low_pass_filter(signal, fs,
            corner_freq=antialias_freq, stop_tol=10)

    # Downsample
    signal, fs_out = filters.downsample(signal, fs, fs_ds)

    # Remove Drift and DC-offset
    if drift_freq is None:
        signal = filters.high_pass_filter(signal, fs_out, corner_freq=drift_freq, stop_tol=10)
        signal = signal - signal.mean(axis=0)

    return signal, fs_out
