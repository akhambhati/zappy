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
    Fit a LocalOutlierFactor model using signal channels as features. Returns an
    outlier score for each channel. 

    Parameters
    ----------
    signal: numpy.ndarray, shape: [n_sample x n_chan]
        Multichannel signal in which to detect outliers.

    n_neighbors: float
        Number of neighbors an inlier should have. This value is used to calculate
        each channel's outlier score. 

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
    outlier_score = sp_stats.zscore(-1*clf.negative_outlier_factor_).clip(min=0)
    return outlier_score


def update_outlier_state(
    otl_scores,
    otl_thresh=10,
    otl_rescale=None,
    otl_state=None,
    otl_decay=0.99):

    """
    Update the outlier state vector based on current outlier scores of features:
        X(t) = AX(t-1) + U(t)

    Parameters
    ----------
    otl_scores: numpy.ndarray, shape: [n]
        Continuous-valued vector of scores corresponding to the likelihood(s) that
        one or more features are outliers.

    otl_thresh: float
        Binary threshold used to define whether otl_scores values are
        true outliers.

    otl_rescale: numpy.ndarray(float), shape: [n]
        Multiplier for the otl_score values to accentuate more drastic outlier events
        from more benign ones. Default is scale is 1.

    otl_state: numpy.ndarray, shape: [n]
        Continuous-valued state vector X(t-1) corresponding to each feature's
        previous outlier state.

    otl_decay: float
        Weighting (A) given to prior outlier states X(t-1). Greater value 
        associated with longer memory and slower exponential decay of
        outlier state. Should not exceed value of 1.

    Returns
    -------
    otl_state: numpy.ndarray, shape: [n]
        Continuous-valued state vector X(t) corresponding to each feature's
        current outlier state.
    """

    # Check that state vector won't explode.
    assert otl_decay <= 1

    # Binarize the outlier scores to determine whether an outlier is observed.
    if otl_thresh is None:
        binary_otl = otl_scores
    else:
        binary_otl = otl_scores > otl_thresh

    # Multiply the outlier scores by a rescale factor
    if otl_rescale is not None:
        binary_otl *= otl_rescale

    # Propagate outlier state forward
    if otl_state is None:
        otl_state = binary_otl
    else:
        otl_state = \
            otl_decay*otl_state + binary_otl

    return otl_state
