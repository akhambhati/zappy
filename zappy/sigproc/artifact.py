# -*- coding: utf-8 -*-
"""
Artifact Rejection Pipelines

:Author: Ankit N. Khambhati
"""


import numpy as np
import scipy.stats as sp_stats
from sklearn.neighbors import LocalOutlierFactor

from .normalize import welford_stats


def population_linelength(signal):
    return np.sqrt(np.sum(np.mean(np.diff(signal, axis=0)**2, axis=0)))


def local_outlier_factor(signal, n_neighbors=None):
    """
    Spatial correlation along a given axis.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that are to be
        normalized. Normalizes along the first axis.

    Returns
    -------
    outlier_score: numpy.ndarray, shape: [n_chan]
        Outlier factor for each channel quantifying the distance of the channel
        from its nearest neighbors within the feature space. Greater values
        are associated with increased outlier likelihood.
    """

    artifact_feats = signal.copy()
    artifact_feats = artifact_feats.T

    clf = LocalOutlierFactor(n_neighbors=(artifact_feats.shape[0] // 4) if n_neighbors is None else n_neighbors)
    outlier = clf.fit_predict(artifact_feats)
    outlier_score = -1*clf.negative_outlier_factor_
    return outlier_score


def check_channel_state(
    signal,

    channel_otl_state=None,
    channel_otl_weight=0.99,
    channel_otl_thresh=10,
    channel_neighbors=10):

    """
    Spatial correlation along a given axis.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that are to be
        normalized. Normalizes along the first axis.

    Returns
    -------
    outlier_score: numpy.ndarray, shape: [n_chan]
        Outlier factor for each channel quantifying the distance of the channel
        from its nearest neighbors within the feature space. Greater values
        are associated with increased outlier likelihood.
    """

    # Assess immediate outlier score
    curr_otl_chan = local_outlier_factor(signal, n_neighbors=channel_neighbors) > channel_otl_thresh

    # Propagate channel state forward
    if channel_otl_state is None:
        channel_otl_state = curr_otl_chan
    else:
        channel_otl_state = \
            channel_otl_weight*channel_otl_state + curr_otl_chan

    return channel_otl_state, curr_otl_chan


def check_epoch_state(
    signal,

    epoch_run_otl=None,
    epoch_run_burn=30,
    epoch_otl_state=None,
    epoch_otl_weight=0.99,
    epoch_otl_thresh=10):
    """
    Spatial correlation along a given axis.

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Signal recorded from multiple electrodes that are to be
        normalized. Normalizes along the first axis.

    Returns
    -------
    outlier_score: numpy.ndarray, shape: [n_chan]
        Outlier factor for each channel quantifying the distance of the channel
        from its nearest neighbors within the feature space. Greater values
        are associated with increased outlier likelihood.
    """

    # Assess immeediate population variability
    pop_LL = population_linelength(signal)

    # Propogate population state forward
    epoch_run_otl, pop_LL_zv = welford_stats(
        pop_LL, run_stats=epoch_run_otl,
        burn=epoch_run_burn, zskip=[-epoch_otl_thresh, epoch_otl_thresh])

    # Calculate cooldown
    if epoch_otl_state is None:
        epoch_otl_state = (np.nan_to_num(pop_LL_zv) > epoch_otl_thresh)
    else:
        epoch_otl_state = \
            epoch_otl_weight*epoch_otl_state + (np.nan_to_num(pop_LL_zv) > epoch_otl_thresh)

    return epoch_otl_state, epoch_run_otl, pop_LL_zv
