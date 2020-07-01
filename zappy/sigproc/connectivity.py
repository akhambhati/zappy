# -*- coding: utf-8 -*-
"""
Apply Connectivity Metrics to raw or transformed intracranial EEG.

:Author: Ankit N. Khambhati
"""


import numpy as np
import pyeisen
import pyfftw


def _tdot(X_cn, imag):
    """
    Tensor multiplication between axes in multi-dimensional array

    Parameters
    ----------
    X_cn: numpy.ndarray, shape: [samples x frequencies x channels]
        Multidimensional tensor containing bandlimited, complex signals.

    imag: bool
        Specify whether to consider imaginary or real+imaginary component
        of the dot product.
    """

    n_ts = X_cn.shape[0]

    # Compute the tensordot
    if imag:
        X_tdot = np.abs(
            np.tensordot(X_cn, np.conj(X_cn), axes=((0), (0))).imag / n_ts)
    else:
        X_tdot = np.abs(
            np.tensordot(X_cn, np.conj(X_cn), axes=((0), (0))) / n_ts)

    return X_tdot


def phase_locking_value(signal, cross_freq=False, coherence=False, imag=False):
    """
    Convolve wavelet family with iEEG signal.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_freqs x n_chan]
        Signal recorded from multiple electrodes that are to be
        notch filtered.

    cross_freq: bool
        Whether to retain the cross-frequency interactions, also known as
        the cross-frequency coupling.
        Default is False.

    coherence: bool
        Whether to weight the phase-locking value by the amplitude of the
        frequency component, akin to calculating the signal coherence.
        Default is False.

    imag: bool
        Whether to only utilize the imaginary component of the complex-valued
        signal. Akin to calculating the imaginary phase-locking or
        imaginary coherence.
        Default is False.

    Returns
    -------
    if cross_freq is False:
        X_plv: numpy.ndarray (Complex), shape: [n_freqs x n_chan x n_chan]
            Phase-locking value between channels per frequency.
    else:
        X_plv: numpy.ndarray (Complex), shape: [n_freqs x n_freqs x n_chan x n_chan]
            Phase-locking value between channels and between frequencies.
    """

    # Get axes sizes
    n_ts, n_wv, n_ch = signal.shape

    # Normalize signal to unit magnitude
    X_cn = signal.copy()
    X_a = np.abs(X_cn)
    if not coherence:
        X_cn /= X_a

    # Compute phase-locking-value using tensor manipulation
    X_plv = _tdot(X_cn, imag=imag)

    # Divide by magnitude of the signal, if coherence
    if coherence:
        X_plv /= np.sqrt(np.tensordot(
            np.expand_dims(np.abs(X_a**2).sum(axis=0), axis=0),
            np.expand_dims(np.abs(X_a**2).sum(axis=0), axis=0),
            axes=((0), (0))))

    # Re-arrange tensordot axes
    X_plv = np.transpose(X_plv, (0, 2, 1, 3))

    # Remove off-diagonal entries if not considering cross-frequency coupling
    if not cross_freq:
        X_plv = X_plv[np.arange(n_wv), np.arange(n_wv), :, :]

    return X_plv


def amplitude_correlation(signal, cross_freq=False):
    """
    Convolve wavelet family with iEEG signal.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_freqs x n_chan]
        Signal recorded from multiple electrodes that are to be
        notch filtered.

    cross_freq: bool
        Whether to retain the cross-frequency interactions, also known as
        the cross-frequency coupling.
        Default is False.

    Returns
    -------
    if cross_freq is False:
        X_acr: numpy.ndarray (Complex), shape: [n_freqs x n_chan x n_chan]
            Phase-locking value between channels per frequency.
    else:
        X_acr: numpy.ndarray (Complex), shape: [n_freqs x n_freqs x n_chan x n_chan]
            Phase-locking value between channels and between frequencies.
    """

    # Get axes sizes
    n_ts, n_wv, n_ch = signal.shape

    # Normalize signal to unit magnitude
    X_cn = signal.copy()
    X_a = np.abs(X_cn)
    X_a = (X_a - np.nanmean(X_a, axis=0)) / np.nanstd(X_a, axis=0)

    # Compute amplitude correlation using tensor manipulation
    X_acr = _tdot(X_a, imag=False)

    """
    # Divide by magnitude of the signal
    X_acr /= np.sqrt(np.tensordot(
        np.expand_dims(np.abs(X_a**2).sum(axis=0), axis=0),
        np.expand_dims(np.abs(X_a**2).sum(axis=0), axis=0),
        axes=((0), (0))))
    """

    # Re-arrange tensordot axes
    X_acr = np.transpose(X_acr, (0, 2, 1, 3))

    # Remove off-diagonal entries if not considering cross-frequency coupling
    if not cross_freq:
        X_acr = X_acr[np.arange(n_wv), np.arange(n_wv), :, :]

    return X_acr
