# -*- coding: utf-8 -*-
"""
Simple visualizer of intracranial EEG signals. Nothing fancy.

:Author: Ankit N. Khambhati
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_time_stacked(sig, fs, wsize=10.0, color='k', labels=None, ax=None):
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

    sig_Z = (sig - np.nanmean(sig, axis=0)) / np.nanstd(sig, axis=0)

    offset = np.arange(n_ch) * 3

    for ch, sig_ch in enumerate(sig_Z.T):
        ax.plot(ts, sig_ch + offset[ch], color=color, alpha=0.5, linewidth=0.5)

        ax.hlines(
            offset[ch], ts[0], ts[-1], color='k', alpha=1.0, linewidth=0.2)

    ax.set_yticks(offset)
    ax.set_yticklabels(labels)

    ax.set_xlim([ts[0], ts[0] + wsize])
    ax.set_ylim([np.min(offset) - 3, np.max(offset) + 3])

    plt.tight_layout()

    return ax
