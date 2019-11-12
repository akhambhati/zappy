# -*- coding: utf-8 -*-
"""
Normalize or standardize signals with respect to reference distributions.

:Author: Ankit N. Khambhati
"""


import numpy as np


def zscore(data_dict, method='robust', scale=1.4826):
    """
    Z-Score the signal along a given axis.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that are to be
        normalized. Normalizes along the first axis.

    method: 'robust' or 'standard', default is 'robust'
        Robust normalization is based on median and median
        absolute deviation, which is more resistant to outliers.
        Standard normalization is based on mean and standard deviation.

    scale: float
        If using robust normalization, the scale parameter specifies the
        width of the variability. Default value is approximately equal to
        one standard deviation.
    """

    # Get a copy of the signal
    signal = signal.copy()

    # Get signal attributes
    n_s, n_ch = signal.shape

 
    if method == 'robust':
        dev = signal - np.nanmedian(signal, axis=0)
        med_abs_dev = scale * np.nanmedian(np.abs(dev), axis=0)
        signal = dev / med_abs_dev

    if method == 'standard':
        signal -= np.nanmean(signal, axis=0)
        signal /= np.nanstd(signal, axis=0)

    return signal
