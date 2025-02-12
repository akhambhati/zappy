# -*- coding: utf-8 -*-
"""
Apply Connectivity Metrics to raw or transformed intracranial EEG.

:Author: Ankit N. Khambhati
"""


import numpy as np
import numpy.ma as ma

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
            np.tensordot(X_cn, np.conj(X_cn), axes=((0), (0))).imag) / n_ts
    else:
        X_tdot = np.abs(
            np.tensordot(X_cn, np.conj(X_cn), axes=((0), (0)))) / n_ts

    return X_tdot


def spectral_synchrony(signal, metric=None, cross_freq=False, signal_mask=None):
    """
    Compute inter-electrode synchrony of the iEEG signal based on its spectral
    characteristics.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_freqs x n_chan]
        Multi-electrode iEEG Signal that has been transformed to the
        complex-domain (via. Hilbert or Wavelet).

    metric: ['plv', 'iplv', 'ciplv', 'coh', 'icoh', 'pli', 'wpli', 'debpli]
        Must supply a metric for calculating spectral synchrony.
        plv - phase locking value (Bruña, R. et. al., 2018).
        iplv - imaginary phase locking value (Bruña, R. et. al., 2018).
        ciplv - corrected imaginary phase locking value (Bruña, R. et. al., 2018).
        coh - coherence (Bruña, R. et. al., 2018).
        icoh - imaginary coherence (Bruña, R. et. al., 2018).
        wpli - weighted phase lag index (Vinck et al., 2011)
        debwpli - debiased weighted phase lag index (Vinck et al., 2011)

    cross_freq: bool
        Whether to retain the cross-frequency interactions, also known as
        the cross-frequency coupling.
        Default is False.

    signal_mask: numpy.ndarray, dtype: bool, shape: [n_sample x n_freqs x n_chan]
        Masking array where indices to be ignored contain TRUE.

    Returns
    -------
    if cross_freq is False:
        X: numpy.ndarray (Complex), shape: [n_freqs x n_chan x n_chan]
            Synchrony between channels per frequency.
    else:
        X: numpy.ndarray (Complex), shape: [n_freqs x n_freqs x n_chan x n_chan]
            Synchony between channels and between frequencies.
    """

    # Get axes sizes
    n_ts, n_wv, n_ch = signal.shape

    X_cn = signal.copy()
    if signal_mask is not None:
        X_cn = ma.array(X_cn, mask=signal_mask)

    # Pre-compute the  magnitude
    X_a = np.abs(X_cn)

    # Compute numerator
    if metric in ['plv', 'iplv', 'ciplv']:
        E_XY_tdot = np.tensordot(X_cn/X_a, np.conj(X_cn/X_a), axes=((0), (0))) / n_ts
    elif metric in ['coh', 'icoh', 'wpli']:
        E_XY_tdot = np.tensordot(X_cn, np.conj(X_cn), axes=((0), (0))) / n_ts
    elif metric in ['pli']:
        E_XY_tdot = np.zeros((n_wv, n_ch, n_wv, n_ch))
        for x_cn in X_cn:
            E_XY_tdot += np.sign(np.tensordot(
                    np.expand_dims(x_cn, axis=0),
                    np.expand_dims(np.conj(x_cn), axis=0),
                    axes=((0), (0))).imag)
        E_XY_tdot /= n_ts
    elif metric in ['debwpli']:
        pass
    else:
        raise NotImplemented('Provided metric, {:s}, is unsupported.'.format(metric))

    # Compute denominator
    if metric in ['plv', 'iplv', 'pli']:
        E_XX_tdot = np.ones_like(E_XY_tdot)
    elif metric in ['ciplv']:
        E_XX_tdot = np.sqrt(1 - np.abs((E_XY_tdot.real) / n_ts)**2)
    elif metric in ['coh', 'icoh']:
        E_XX_tdot = np.sqrt(np.tensordot(
            np.expand_dims((X_a**2).mean(axis=0), axis=0),
            np.expand_dims((X_a**2).mean(axis=0), axis=0),
            axes=((0), (0))))
    elif metric in ['wpli']:
        E_XX_tdot = np.zeros((n_wv, n_ch, n_wv, n_ch))
        for x_cn in X_cn:
            E_XX_tdot += np.abs(np.tensordot(
                    np.expand_dims(x_cn, axis=0),
                    np.expand_dims(np.conj(x_cn), axis=0),
                    axes=((0), (0))).imag)
        E_XX_tdot /= n_ts
    elif metric in ['debwpli']:
        pass

    # Compute final metric
    if metric in ['iplv', 'ciplv', 'icoh', 'wpli']:
        E_XY_tdot = np.abs(E_XY_tdot.imag / E_XX_tdot)
    elif metric in ['plv', 'coh']:
        E_XY_tdot = np.abs((E_XY_tdot / E_XX_tdot))
    elif metric in ['pli']:
        E_XY_tdot = np.abs(E_XY_tdot / E_XX_tdot)
    elif metric in ['debwpli']:
        sum_im_csd = np.zeros((n_wv, n_ch, n_wv, n_ch))
        sum_abs_im_csd = np.zeros((n_wv, n_ch, n_wv, n_ch))
        sum_sq_im_csd = np.zeros((n_wv, n_ch, n_wv, n_ch))

        for x_cn in X_cn:
            tdot = np.tensordot(
                    np.expand_dims(x_cn, axis=0),
                    np.expand_dims(np.conj(x_cn), axis=0),
                    axes=((0), (0))).imag
            sum_im_csd += tdot
            sum_abs_im_csd += np.abs(tdot)
            sum_sq_im_csd += np.abs(tdot**2)

        numer = sum_im_csd**2 - sum_sq_im_csd
        denom = sum_abs_im_csd**2 - sum_sq_im_csd
        z_denom = denom == 0
        denom[z_denom] = 1
        E_XY_tdot = numer / denom
        E_XY_tdot[z_denom] = 0
    else:
        return None
    E_XY_tdot = np.nan_to_num(E_XY_tdot)

    # Rearrange the tensor and remove cross-frequency components if desired
    E_XY_tdot =  np.transpose(E_XY_tdot, (0, 2, 1, 3))

    # Remove off-diagonal entries if not considering cross-frequency coupling
    if not cross_freq:
        E_XY_tdot = E_XY_tdot[np.arange(n_wv), np.arange(n_wv), :, :]

    return E_XY_tdot


def amplitude_correlation(signal, cross_freq=False):
    """
    Compute inter-electrode amplitude correlation of the iEEG signal.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_freqs x n_chan]
        Multi-electrode iEEG Signal that has been transformed to the
        complex-domain (via. Hilbert or Wavelet), or transformed to the
        real-valued analytic signal.

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


def xcorr_fft(signal, fs):
    """
    Compute inter-electrode cross-correlation of the iEEG signal.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan x n_chan]
        Multi-electrode iEEG signal.

    fs: float
        Sampling frequency of the signal.

    Returns
    -------
    xcr: numpy.ndarray, shape: [n_lags x n_chan x n_chan]
        Peak magnitude cross-correlation between channels.

    lags: numpy.ndarray, shape: [n_lags]
        Delay of the cross-correlation between channels in seconds.
    """

    # Get data attributes
    n_samp, n_chan = signal.shape

    # Calibrate lags
    lags = np.arange(-n_samp // 2+1, n_samp // 2+1) / fs
    n_lags = len(lags)

    # Pre-compute channel-wise FFT
    FFT_X = np.fft.fft(signal, axis=0)

    # Pre-compute reverse channel-wise FFT
    FFTUD_Y = np.fft.fft(np.flipud(signal), axis=0)

    # Form a grid for cross-channel FFT multiplication
    GRID = np.meshgrid(
            np.arange(n_chan),
            np.arange(n_chan))
    FFT_GRID = FFT_X[:,GRID[0]] * FFTUD_Y[:,GRID[1]]

    # Compute the inverse FFT of the cross-channel product
    xcr = np.fft.fftshift(np.real(np.fft.ifft(FFT_GRID, axis=0)), axes=0)
    return xcr, lags


def xcorr_mag(signal, fs, tau_min=0, tau_max=None):
    """
    Compute inter-electrode cross-correlation of the iEEG signal.

    XC = max(abs(xcorr(x1, x2)))
    delay = argmax(abs(xcorr(x1, x2)))

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_freqs x n_chan]
        Multi-electrode iEEG Signal that has been transformed to its
        analytic amplitude.

    fs: float
        Sampling frequency of the signal.

    tau_min: float
        Shortest latency to consider in the cross-correlation window estimate

    tau_max: float
        Longest latency to consider in the cross-correlation window estimate

    Returns
    -------
    xcr: numpy.ndarray, shape: [n_freqs x n_chan x n_chan]
        Peak magnitude cross-correlation between channels per frequency.

    delay: numpy.ndarray, shape: [n_freqs x n_chan x n_chan]
        Delay of the peak magnitude cross-correlation between channels per frequency.
    """

    # Get data attributes
    n_samp, n_freq, n_chan = signal.shape
    triu_ix, triu_iy = np.triu_indices(n_chan, k=1)

    # Normalize the signal
    signal = np.abs(signal)
    signal -= signal.mean(axis=0)
    signal /= signal.std(axis=0)

    # Initialize adjacency matrix
    adj = np.zeros((n_freq, n_chan, n_chan))
    delay = np.zeros((n_freq, n_chan, n_chan))

    if tau_max is None:
        tau_max = n_samp / fs
    lags = np.hstack((range(0, n_samp, 1),
                      range(-n_samp, 0, 1))) / fs
    tau_ix = np.flatnonzero((np.abs(lags) >= tau_min) &
                            (np.abs(lags) <= tau_max))

    # Use FFT to compute cross-correlation
    sig_fft = np.fft.rfft(
        np.vstack((signal, np.zeros_like(signal))),
        axis=0)

    # Iterate over all edges
    for n1, n2 in zip(triu_ix, triu_iy):
        xc = 1 / n_samp * np.fft.irfft(
            sig_fft[:, :, n1] * np.conj(sig_fft[:, :, n2]), axis=0)
        adj[:, n1, n2] = np.max(np.abs(xc[tau_ix]), axis=0)

        opt_lag = lags[tau_ix][np.argmax(np.abs(xc[tau_ix]), axis=0)]
        delay[:, n1, n2] = opt_lag
    adj = adj + adj.transpose((0,2,1))
    delay = delay + -1*delay.transpose((0,2,1))

    return adj, delay


def gcc_phat(signal, fs):
    """
    Compute inter-electrode cross-correlation of the iEEG signal.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan x n_chan]
        Multi-electrode iEEG signal.

    fs: float
        Sampling frequency of the signal.

    Returns
    -------
    xcr: numpy.ndarray, shape: [n_lags x n_chan x n_chan]
        Peak magnitude cross-correlation between channels.

    lags: numpy.ndarray, shape: [n_lags]
        Delay of the cross-correlation between channels in seconds.
    """

    # Get data attributes
    n_samp, n_chan = signal.shape

    # Calibrate lags
    lags = np.arange(-n_samp // 2+1, n_samp // 2+1) / fs
    n_lags = len(lags)

    # Pre-compute channel-wise FFT
    FFT_X = np.fft.fft(signal, axis=0)

    # Form a grid for cross-channel FFT multiplication
    GRID = np.meshgrid(
            np.arange(n_chan),
            np.arange(n_chan))
    FFT_GRID = FFT_X[:,GRID[0]] * np.conj(FFT_X[:,GRID[1]])

    # Compute the inverse FFT of the cross-channel product
    xcr = np.fft.fftshift(np.real(np.fft.ifft(FFT_GRID / np.abs(FFT_GRID), axis=0)), axes=0)
    return xcr, lags

