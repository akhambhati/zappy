# -*- coding: utf-8 -*-
"""
Apply Wavelet Transform to intracranial EEG.

:Author: Ankit N. Khambhati
"""


import numpy as np
import pyeisen
import pyfftw


def _reserve_fftw_mem(kernel_len, signal_len, n_kernel, threads=6):
    a = pyfftw.empty_aligned(
        (kernel_len + signal_len, n_kernel), dtype=complex)
    fft = pyfftw.builders.fft(a, axis=0, threads=threads)
    ifft = pyfftw.builders.ifft(a, axis=0, threads=threads)

    return fft, ifft


def gen_morlet_family(Fs, FQ_LOW=3, FQ_HIGH=150, FQ_N=50, CYC_N=6, logspace=True):
    """
    Generate a whole family of Morlet wavelets.

    Parameters
    ----------
    Fs: float
        Sampling frequency of the iEEG signal to which the wavelet will be applied.

    FQ_LOW: float
        Lower frequency bound of the wavelets.

    FQ_HIGH: float
        igher frequency bound of the wavelets.

    FQ_N: int
        Number of wavelets to generate within the frequency range.

    CYC_N: int
        Number of cycles per wavelet.

    logspace: bool
        Frequency spacing of the wavelets, if True then frequencies are log-sapced.
    """

    if logspace:
        wv_freqs = np.logspace(np.log10(FQ_LOW), np.log10(FQ_HIGH), FQ_N)
    else:
        wv_freqs = np.linspace(FQ_LOW, FQ_HIGH, FQ_N)
    wv_cycles = np.ones(len(wv_freqs))*CYC_N

    family = pyeisen.family.morlet(freqs=wv_freqs, cycles=wv_cycles, Fs=Fs)

    # Remove the mean of each wavelet kernel
    family['kernel'] = (family['kernel'].T - np.nanmean(family['kernel'], axis=1)).T

    return family


def convolve_family(signal, family, mem_fft=True, interp_nan=True):
    """
    Convolve wavelet family with iEEG signal.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that are to be
        notch filtered.

    mem_fft: bool
        Reserve memory for convolution. Recommended if signals are long and if
        transformation will be iterated over multiplee signals of the
        same length. Default is True.

    interp_nan: bool
        Interpolate NaNs in the signal. Default is True.

    Returns
    -------
    signal: numpy.ndarray (Complex), shape: [n_sample x n_freqs x n_chan]
        Complex-valued, wavelet-transformed signal.
    """

    n_s, n_c = signal.shape
    n_k, n_w = family['kernel'].shape

    # Handle FFT, IFFT definition if reserving memory
    fft, ifft = [None, None]
    if mem_fft:
        print('- Reserving memory for wavelet convolution')
        fft, ifft = _reserve_fftw_mem(
            kernel_len=n_w,
            signal_len=n_s,
            n_kernel=n_k)

    # Setup signal
    wv_signal = np.zeros((n_s, n_k, n_c),
                         dtype=np.complex)

    # Iterate over each channel and convolve
    print('- Iteratively convolving wavelet with each channel')
    for ch_ii in range(n_c):
        print('    - {} of {}'.format(ch_ii + 1, n_c))

        out = pyeisen.convolve.fconv(
                family['kernel'][:, :].T,
                signal[:, ch_ii].reshape(-1, 1),
                fft=fft, ifft=ifft, interp_nan=interp_nan)

        wv_signal[:, :, ch_ii] = out[:, :, :][:, :, 0]

    return wv_signal
