"""
Utilities to work with the signal data_dict.

Author: Ankit N. Khambhati
Last Updated: 2019/03/17
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig


def common_avg_reref(data_dict, channel_dist=None, channel_group=None):
    """Re-reference the signal array to the (weighted) common average.

    Parameters
    ----------
        channel_dist: numpy.ndarray, shape: [n_chan x n_chan]
            Array specifying inter-channel distances
            (e.g. euclidean, geodesic, etc). If None, no distance-weighting is
            performed.
    """

    pivot_axis(data_dict, 'channel')

    if channel_group is None:
        channel_group = np.ones(data_dict['signal'].shape[0])
    assert data_dict['signal'].shape[0] == len(channel_group)

    for grp_id in np.unique(channel_group):
        grp_ix = np.flatnonzero(channel_group == grp_id)

        if type(channel_dist) == np.ndarray:
            channel_dist_grp = channel_dist[grp_ix, :][:, grp_ix]

            chan_prox = 1 / (channel_dist_grp)
            chan_prox[np.isinf(chan_prox)] = 0
            chan_prox /= chan_prox.sum(axis=1)
            common = np.tensordot(chan_prox, data_dict['signal'][grp_ix], axes=1)
        else:
            common = data_dict['signal'][grp_ix].mean(axis=0)
        data_dict['signal'][grp_ix] -= common

    pivot_axis(data_dict, 'sample')


def decimate(data_dict, fs_new):
    """Decimate the signal array, and anti-alias filter."""

    ts_ax = get_axis(data_dict, 'sample')
    fs = get_fs(data_dict)

    q = int(np.round(fs / fs_new))
    fs = fs / q

    data_dict['signal'] = sig.decimate(
        data_dict['signal'], q=q, ftype='fir', zero_phase=True, axis=ts_ax)

    n_ts = data_dict['signal'].shape[ts_ax]
    for subkey in data_dict['sample']:
        data_dict['sample'][subkey] = data_dict['sample'][subkey][::q]


def notchline(data_dict, freq_list, bw=2, harm=True):
    """Notch filter the line noise and harmonics"""

    ts_ax = get_axis(data_dict, 'sample')

    fs = get_fs(data_dict)
    nyq_fs = fs / 2

    freq_list = np.unique(freq_list)
    freq_list = freq_list[freq_list > 0]
    freq_list = freq_list[freq_list < nyq_fs]

    for ff in freq_list:
        if (ff + bw) >= nyq_fs:
            continue
        b, a = sig.iirnotch(ff / nyq_fs, ff / bw)
        data_dict['signal'] = sig.filtfilt(
            b, a, data_dict['signal'], axis=ts_ax)


def zscore(data_dict, lbl, method='robust', scale=1.4826):
    """Z-Score the signal along the provided label"""

    pivot_axis(data_dict, lbl)

    if method == 'robust':
        dev = data_dict['signal'] - np.nanmedian(data_dict['signal'], axis=0)
        med_abs_dev = scale * np.nanmedian(np.abs(dev), axis=0)
        data_dict['signal'] = dev / med_abs_dev

    if method == 'standard':
        data_dict['signal'] -= np.nanmean(data_dict['signal'], axis=0)
        data_dict['signal'] /= np.nanstd(data_dict['signal'], axis=0)


def plot_time_stacked(data_dict, ax):
    """Plot of the normalized signal in a stacked montage."""

    sig = data_dict['signal'][...]
    sig_Z = (sig - np.nanmean(sig, axis=0)) / np.nanstd(sig, axis=0)

    offset = np.arange(sig_Z.shape[1]) * 3

    for ch, sig_ch in enumerate(sig_Z.T):
        ax.plot(
            data_dict['sample']['timestamp'],
            sig_ch + offset[ch],
            color='b',
            alpha=0.5,
            linewidth=0.5)

        ax.hlines(
            offset[ch],
            data_dict['sample']['timestamp'][0],
            data_dict['sample']['timestamp'][-1],
            color='k',
            alpha=0.5,
            linewidth=0.1)

    ax.set_yticks(offset)
    ax.set_yticklabels(data_dict['channel']['label'])

    ax.set_xlim([
        data_dict['sample']['timestamp'][0],
        data_dict['sample']['timestamp'][0] + 10
    ])
    #    ax.set_ylim([
    return ax
