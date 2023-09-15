# -*- coding: utf-8 -*-
"""
Normalize or standardize signals with respect to reference distributions.

:Author: Ankit N. Khambhati
"""


import numpy as np
import pandas as pd


def zscore(signal, method='robust', scale=1.4826):
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


def welford_stats(next_obs, run_stats=None, burn=10, zskip=[-np.inf, np.inf],
                  norm='znorm'):
    if run_stats is None:
        run_stats = {
                'N': 0,
                'mean': np.zeros_like(next_obs), 
                'M2': np.zeros_like(next_obs)}

    if run_stats['N'] > burn:
        stdv = np.sqrt(run_stats['M2'] / run_stats['N'])
        if (stdv == 0).any():
            next_obs_zs = np.nan*np.zeros_like(next_obs)
        else:
            if norm=='znorm':
                next_obs_zs = (next_obs - run_stats['mean']) / stdv
            else:
                next_obs_zs = next_obs / (stdv**2)
        if ((next_obs_zs < zskip[0]).any() or (next_obs_zs > zskip[1]).any()):
            return run_stats, next_obs_zs
    else:
        next_obs_zs = np.nan*np.zeros_like(next_obs)

    run_stats['N'] += 1

    delta = next_obs - run_stats['mean']
    run_stats['mean'] += delta / run_stats['N']

    delta2 = next_obs - run_stats['mean']
    run_stats['M2'] += delta * delta2

    return run_stats, next_obs_zs


def running_zscore(signal, fs, win):
    """
    Z-Score the signal along a given axis.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that are to be
        normalized. Normalizes along the first axis.

    fs: float
        Sampling frequency of signal.

    win: float
        Historical window to z-score against.
    """

    # Get a copy of the signal
    signal = signal.copy()

    # Get signal attributes
    n_s, n_ch = signal.shape

    # Convert to pandas
    df = pd.DataFrame(signal)
    win_samp = int(fs*win)

    r = df.rolling(window=win_samp)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (df-m)/s

    return z.values


def smoothing(signal, fs, win):
    """
    Smooth the signal.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that are to be
        normalized. Normalizes along the first axis.

    fs: float
        Sampling frequency of signal.

    win: float
        Historical window to z-score against.
    """

    # Get a copy of the signal
    signal = signal.copy()

    # Get signal attributes
    n_s, n_ch = signal.shape

    # Convert to pandas
    df = pd.DataFrame(signal)
    win_samp = int(fs*win)

    r = df.rolling(window=win_samp)
    m = r.mean()

    return m.values
