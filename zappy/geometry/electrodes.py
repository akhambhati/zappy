"""electrode.py
Utilities to define and manipulate electrode data.

Author: Ankit N. Khambhati
Last Updated: 2023/09/11
"""

import os
from glob import glob
from tqdm import tqdm

from typing import Any
from typing import Dict
from typing import Tuple
from typing import List
from typing import TypedDict

# Third-Party Packages
import numpy as np
import numpy.typing as npt
import pandas as pd

from .spatial import electrode_distance


class ElectrodeContact(TypedDict):
    """Defines the attributes of an electrode contact."""

    name: str
    index: int
    coord: npt.NDArray[np.float_]


class VirtualContact(ElectrodeContact):
    """Defines the attributes of an electrode contact."""

    anode_index: List[int]
    cathode_index: List[int]


class ElectrodeGroup():
    """Defines the attributes of electrode groups."""

    def __init__(self, name: str, type: str, electrode_contacts: List[ElectrodeContact]):
        self.name = name
        self.type = type
        self.electrode_contacts = electrode_contacts

    @property
    def intercontact_distance(self):
        """Reflect the distances between contacts."""
        return electrode_distance(np.array([contact['coord'] for contact in self.electrode_contacts]))

    @property
    def indices(self):
        """Indices of all the member contacts."""
        return np.array([contact['index'] for contact in self.electrode_contacts])

    @property
    def coords(self):
        """Indices of all the member contacts."""
        return np.array([contact['coord'] for contact in self.electrode_contacts])

                 
def closest_square(n):
    n = int(n)
    i = int(np.ceil(np.sqrt(n)))
    while True:
        if (n % i) == 0:
            break
        i += 1
    assert n == (i * (n // i))
    return i, n // i


def make_virtual_bipolar(electrode_groups: List[ElectrodeGroup]) -> List[ElectrodeGroup]:
    vgroups = []
    vidx = 0
    for egrp in electrode_groups:
        vname = 'v{}'.format(egrp.name)
        vtype = 'v{}'.format(egrp.type)
        n_contact = len(egrp.electrode_contacts)
        if 'grid' == egrp.type:
            n_row, n_col = closest_square(n_contact)
        else:
            n_row, n_col = [n_contact, 1]

        CA = np.arange(n_contact).reshape((n_row, n_col), order='F')
        vchans = []
        if n_row > 1:
            for ii, (bp1, bp2) in enumerate(zip(CA[:-1, :].flatten(), CA[1:, :].flatten())):
                vchan: VirtualContact = {'name': 'v{}{}'.format(egrp.name, ii+1), 'index': vidx, 'anode_index': [egrp.electrode_contacts[bp1]], 'cathode_index': [egrp.electrode_contacts[bp2]], 'coord': egrp.coords[[bp1, bp2]].mean(axis=0)}
                vchans.append(vchan) 
                vidx += 1

        if n_col > 1:
            for ii, (bp1, bp2) in enumerate(zip(CA[:, :-1].flatten(), CA[:, 1:].flatten())):
                vchan: VirtualContact = {'name': 'v{}{}'.format(egrp.name, ii+1), 'index': vidx, 'anode_index': [egrp.electrode_contacts[bp1]], 'cathode_index': [egrp.electrode_contacts[bp2]], 'coord': egrp.coords[[bp1, bp2]].mean(axis=0)}
                vchans.append(vchan) 
                vidx += 1

        vgrp: ElectrodeGroup = {'name': vname, 'type': vtype, 'electrode_contacts': vchans}
        vgroups.append(vgrp)

    return vgroups
