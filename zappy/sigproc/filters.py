# -*- coding: utf-8 -*-
"""
Commonly used low-pass, high-pass, and notch filtering schemes
for intracranial EEG processing.

:Author: Ankit N. Khambhati
"""


import numpy as np
import scipy.signal as sig


def decimate(signal, fs_old, fs_new):
    """
    Apply an anti-alias filter and downsample the signal.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that are to be
        low-pass filtered and then decimated.

    fs_old: float
        Sampling frequency of the signal, before decimation.

    fs_new: float
        Target sampling frequency of the signal, after decimation. Actual
        frequency of downsampled signal will vary, refer to the output fs
        for most precise frequency.

    Returns
    -------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Decimated signal.

    q: int
        Decimation factor by which the signal was downsampled.
    """

    # Get a copy of the signal
    signal = signal.copy()

    # Get signal attributes
    n_s, n_ch = signal.shape

    # Decimation factor
    q = int(np.round(fs_old / fs_new))

    # Re-derived precise sampling frequency after finding optimal q
    fs_out = fs_old / q

    # Apply decimation to the signal, along the time axis
    signal = sig.decimate(
        signal, q=q, ftype='fir', zero_phase=True, axis=0)

    return signal, q


def remove_drift(signal, fs):
    """
    Apply a high-pass Butterworth filter to remove drift component. Should
    preserve Delta frequencies >1 Hz.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that are to be
        notch filtered.

    fs: float
        Sampling frequency of the signal.

    Returns
    -------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal with drift removed.
    """

    # Get a copy of the signal
    signal = signal.copy()

    # Get signal attributes
    n_s, n_ch = signal.shape

    # Get butterworth filter parameters
    buttord_params = {'wp': 1.0,            # Passband 1 Hz
                      'ws': 0.5,            # Stopband 0.5 Hz
                      'gpass': 3,           # 3dB corner at pass band
                      'gstop': 60,          # 60dB min. attenuation at stop band 
                      'analog': False,      # Digital filter
                      'fs': fs}
    ford, wn = sig.buttord(**buttord_params)

    # Design the filter using second-order sections to ensure better stability
    sos = sig.butter(ford, wn, btype='highpass', output='sos', fs=fs)

    # Apply zero-phase forward/backward filter signal along the time axis
    signal = sig.sosfiltfilt(sos, signal, axis=0)

    return signal


def notch_line(signal, fs, notch_freq=60.0, bw=2.0, harm=True):
    """
    Apply a Notch filter at a specified Notch frequency (and harmonics).

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that are to be
        notch filtered.

    fs: float
        Sampling frequency of the signal.

    notch_freq: float, default = 60
        The fundamental frequency to notch filter.

    bw: float, default = 2.0
        Bandwidth to filter around notch frequency,
        [notch_freq-bw, notch_freq+bw]

    harm: bool
        If true, then notch all harmonics at specified bandwidth, upto the
        Nyquist frequency.

    Returns
    -------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Notch filtered signal.
    """

    # Get a copy of the signal
    signal = signal.copy()

    # Get signal attributes
    n_s, n_ch = signal.shape

    # Get nyquist
    nyq_fs = fs / 2

    # Generate a list of frequencies
    if harm:
        freq_list = np.arange(notch_freq, nyq_fs, notch_freq)
    else:
        freq_list = np.array([notch_freq])

    # Remove frequencies that are near the nyquist edge
    freq_list = freq_list[(freq_list-bw) > 0]
    freq_list = freq_list[(freq_list+bw) < nyq_fs]

    # Iteratively notch each of the frequencies in freq_list
    for ff in freq_list:

        # Design the filter for each frequency range in freq_list
        b, a = sig.iirnotch(ff, ff / bw, fs=fs)

        # Apply a zero-phase, forward/backward filter to signal along time axis
        signal = sig.filtfilt(b, a, signal, axis=0)

    return signal
