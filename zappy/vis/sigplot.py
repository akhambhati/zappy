# -*- coding: utf-8 -*-
"""
Simple visualizer of intracranial EEG signals. Nothing fancy.

:Author: Ankit N. Khambhati
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_time_stacked(sig, fs, wsize=10.0, color='k', labels=None, zscore=True, scale=3, ax=None):
    """
    Plot of the normalized signal in a stacked montage.

    Parameters
    ----------
    sig: np.ndarray, shape: [n_sample, n_ch]
        Time series signal.

    fs: float
        Sampling frequency of the signal (in Hz)

    wsize: float
        Window size in seconds.

    color: str
        Color of the plot lines.

    labels: array-like, len(n_ch)
        Labels corresponding to the channel names.

    scale: float, default=3.0
        Standard deviations of signal fluctuation by which the montage is
        vertically spaced for each channel.

    ax: matplotlib axis
        For updating the plot post-hoc.
    """

    sig = sig[...]
    n_s, n_ch = sig.shape
    ts = np.arange(0, n_s) / fs
    if labels is None:
        labels = ['Ch{}'.format(ix + 1) for ix in range(n_ch)]

    if ax is None:
        plt.figure(figsize=(24, 12))
        ax = plt.subplot(111)

    if zscore:
        sig_Z = (sig - np.nanmean(sig, axis=0)) / np.nanstd(sig, axis=0)
    else:
        sig_Z = sig

    offset = np.arange(n_ch) * scale

    for ch, sig_ch in enumerate(sig_Z.T):
        ax.plot(ts, sig_ch + offset[ch], color=color, alpha=0.5, linewidth=0.5)

        ax.hlines(
            offset[ch], ts[0], ts[-1], color='k', alpha=1.0, linewidth=0.2)

    ax.set_yticks(offset)
    ax.set_yticklabels(labels)

    ax.set_xlim([ts[0], ts[0] + wsize])
    ax.set_ylim([np.min(offset) - scale, np.max(offset) + scale])

    plt.tight_layout()

    return ax


def plot_spectrogram(sig, fs, freqs, wsize=10.0, scale=3, ax=None):
    sig = sig[...]
    n_s, n_f = sig.shape

    ts = np.arange(0, n_s) / fs

    if ax is None:
        plt.figure(figsize=(24, 12))
        ax = plt.subplot(111)

    sig_Z = (sig - np.nanmean(sig, axis=0)) / np.nanstd(sig, axis=0)

    ax.imshow(sig_Z.T,
       extent=[ts[0], ts[-1], np.log10(freqs[-1]), np.log10(freqs[0])],
       vmin=-scale, vmax=scale,
       aspect='auto',
       cmap='RdBu_r')

    ax.set_yticks(np.log10(np.array([1, 3, 8, 15, 30, 70, 170])))
    ax.set_yticklabels(np.array([1, 3, 8, 15, 30, 70, 170]))    

    if wsize is not None:
        ax.set_xlim([ts[0], ts[0] + wsize])

    return ax


def plot_heatmap(sig, fs, wsize=10.0, labels=None, scale=3, ax=None):
    sig = sig[...]
    n_s, n_c = sig.shape

    ts = np.arange(0, n_s) / fs

    if ax is None:
        plt.figure(figsize=(24, 12))
        ax = plt.subplot(111)

    ax.imshow(sig[:, ::-1].T,
       extent=[ts[0], ts[-1], 0, n_c],
       vmin=-scale, vmax=scale, interpolation='none',
       aspect='auto', cmap='RdBu_r')

    ax.set_yticks(np.arange(n_c) + 0.5)
    if labels is None:
        ax.set_yticklabels(np.arange(n_c))
    else:
        ax.set_yticklabels(labels)

    if wsize is not None:
        ax.set_xlim([ts[0], ts[0] + wsize])

    return ax


def plot_heatmap_raw(sig, fs, wsize=10.0, labels=None, tail_cutoff=0, cmap='RdBu_r', ax=None):
    sig = sig[...]
    n_s, n_c = sig.shape

    ts = np.arange(0, n_s) / fs

    if ax is None:
        plt.figure(figsize=(24, 12))
        ax = plt.subplot(111)

    vmin = np.nanpercentile(sig, tail_cutoff)
    vmax = np.nanpercentile(sig, 100-tail_cutoff)
    mat = ax.imshow(sig[:, ::-1].T,
       extent=[ts[0], ts[-1], 0, n_c],
       vmin=vmin, vmax=vmax, interpolation='none',
       aspect='auto', cmap=cmap)
    plt.colorbar(mat, ax=ax)

    ax.set_yticks(np.arange(n_c) + 0.5)
    if labels is None:
        ax.set_yticklabels(np.arange(n_c))
    else:
        ax.set_yticklabels(labels)

    if wsize is not None:
        ax.set_xlim([ts[0], ts[0] + wsize])

    return ax
