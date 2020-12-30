# -*- coding: utf-8 -*-
"""
Re-referencing of intracranial EEG signals according to different schemes.

:Author: Ankit N. Khambhati
"""

import numpy as np


def general_reref(signal, channel_dist=None, channel_group=None):
    """
    Re-reference the signal array to the (weighted) common average by
    electrode groups. Flexible inputs enable different forms of re-referencing,
    including common average, Laplacian, bipolar, GM/WM, surface/depth.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that are to be
        re-referenced to a variation of the common average reference scheme.

    channel_dist: numpy.ndarray, shape: [n_chan x n_chan]; default is None
        Array specifying inter-channel distances (e.g. euclidean, geodesic, etc).
        If None, no distance-weighting is performed.

    channel_group: numpy.ndarray or list: shape: [n_chan]; default is None
        List or array with indicators (numbers, strings) assigning channels
        into groups. If None, all electrodes assigned to a single group.
        One common use is to separate surface and depth electrodes into groups.
    """

    # Get a copy of the signal
    signal = signal.copy()

    # Get signal dimensionality
    n_s, n_ch = signal.shape

    # If channel_dist not specified, weight all channel distances the same
    if channel_dist is None:
        channel_dist = np.ones((n_ch, n_ch))
        channel_dist[np.diag_indices(n_ch)] = 0

    # If groups not specified, then organize channels into one group
    if channel_group is None:
        channel_group = np.ones(n_ch)
    assert n_ch == len(channel_group)

    # Iterate over the unique groups of channels
    for grp_id in np.unique(channel_group):
        grp_ix = np.flatnonzero(channel_group == grp_id)

        # Reference based on weighted inter-channel distances within group
        if type(channel_dist) == np.ndarray:
            channel_dist_grp = channel_dist[grp_ix, :][:, grp_ix]

            # Compute distance to proximity
            chan_prox = 1 / (channel_dist_grp)
            chan_prox[np.isinf(chan_prox)] = 0

            # Normalize proximity to sum to one (constrained weighted average)
            chan_prox /= chan_prox.sum(axis=1)
            common = np.tensordot(chan_prox, signal[:, grp_ix].T, axes=1)
        else:
            common = signal[:, grp_ix].T.mean(axis=0)
        signal[:, grp_ix] = (signal[:, grp_ix].T - common).T

    return signal


def bipolar_reref(signal, bipolar_pairs):
    """
    Re-reference the signal array to the (weighted) common average by
    electrode groups. Flexible inputs enable different forms of re-referencing,
    including common average, Laplacian, bipolar, GM/WM, surface/depth.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that are to be
        re-referenced to a variation of the common average reference scheme.

    bipolar_pairs: numpy.ndarray, shape: [n_pair x 2]
        Array specifying indices for the anode and cathode in each bipolar
        referenc pair.
    """

    # Get signal dimensionality
    n_s, n_ch = signal.shape

    # Get bipolar dimensionality
    n_pair = bipolar_pairs.shape[0]

    # Construct a new bipolar signal
    signal_bp = np.zeros((n_s, n_pair))

    # Populate bipolar signal
    for p in range(n_pair):
        signal_bp[:, p] = (signal[:, bipolar_pairs[p, 0]] -  
                           signal[:, bipolar_pairs[p, 1]])

    return signal_bp
