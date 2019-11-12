# -*- coding: utf-8 -*-
"""
Functions for handling anatomical information for electrodes.

:Author: Ankit N. Khambhati
"""


import numpy as np


def assign_wm_to_closest_nonwm(atlas_lbl, el_dist):
    wm_ix = np.flatnonzero(atlas_lbl == 'whitematter')

    for ix in wm_ix:
        ix_dist = el_dist[ix, :]

        for lbl in atlas_lbl[np.argsort(ix_dist)]:
            if lbl != 'whitematter':
                sel_lbl = lbl
                break
        atlas_lbl[ix] = sel_lbl

    return atlas_lbl
