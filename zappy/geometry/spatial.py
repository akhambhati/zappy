# -*- coding: utf-8 -*-
"""
Functions for assessing spatial information related to intracranial electrodes.

:Author: Ankit N. Khambhati
"""


import numpy as np


def electrode_distance(xyz_coord):
    """
    Compute the Euclidean distance between channels using localization data.

    Parameters
    ----------
    xyz_coord: np.ndarray, shape: [n_chan x 3]
        Three dimensional for electrodes in millimeters

    Return
    ------
    channel_dist: numpy.ndarray, shape: [n_chan x n_chan]
        Inter-channel distance array.
    """

    # Get number of channels
    n_chan = xyz_coord.shape[0]

    # Construct a meshgrid to efficiently compute all possible distances
    c_i, c_j = np.meshgrid(np.arange(n_chan), np.arange(n_chan))

    # Compute distances using Euclidean geometry
    channel_dist = np.sqrt(
        np.sum((xyz_coord[c_i, :] - xyz_coord[c_j, :])**2, axis=-1))

    return channel_dist
