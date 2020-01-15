"""
Utilities to work with signal data_dict.

Author: Ankit N. Khambhati
Last Updated: 2019/11/19
"""


import numpy as np


def load_data_dict(dd_path):
        df = np.load(dd_path)
        data_dict = {}
        for d in df:
            data_dict[d] = df[d][()]
        return data_dict


def check_dict_layout(data_dict):
    """Check that the data_dict conforms to the basic template."""

    assert 'signal' in data_dict
    assert 'axis_ord' in data_dict

    assert 'sample' in data_dict
    assert 'channel' in data_dict
    assert 'sample' in data_dict['axis_ord']
    assert 'channel' in data_dict['axis_ord']

    assert 'timestamp' in data_dict['sample']
    assert 'label' in data_dict['channel']

    sig_shape = data_dict['signal'].shape
    n_ts = len(data_dict['sample']['timestamp'])
    ax = get_axis(data_dict, 'sample')
    assert n_ts == sig_shape[ax]

    n_ch = len(data_dict['channel']['label'])
    ax = get_axis(data_dict, 'channel')
    assert n_ch == sig_shape[ax]

    for key in data_dict['sample']:
        assert n_ts == len(data_dict['sample'][key])

    for key in data_dict['channel']:
        assert n_ch == len(data_dict['channel'][key])


def get_axis(data_dict, lbl):
    if (type(lbl) != str) or (lbl not in data_dict['axis_ord']):
        raise Exception('Cannot locate axis label `{}`.'.format(lbl))
    return np.flatnonzero(data_dict['axis_ord'] == lbl)[0]


def pivot_axis(data_dict, lbl):
    """Transpose axis with lbl to the first dimension"""

    piv_ax = get_axis(data_dict, lbl)
    all_ax = np.arange(len(data_dict['axis_ord']))
    reord_ax = np.insert(np.delete(all_ax, piv_ax), 0, piv_ax)

    data_dict['axis_ord'] = data_dict['axis_ord'][reord_ax]
    data_dict['signal'] = np.transpose(data_dict['signal'], reord_ax)


def get_fs(data_dict):
    return np.median(1 / np.diff(data_dict['sample']['timestamp']))


def subset(data_dict, **kwargs):
    """
    Retrieve a data_dict copy corresponding to a data subset.

    Parameters
    ----------
        kwargs: keys correspond to axis labels, with contents of index lists or slices

    Return
    ------
        data_dict: a DEEP copy of the data_dict corresponding to the data subset.
    """

    slice_inds = ()
    for k_ii, key in enumerate(data_dict['axis_ord']):
        if key in kwargs:
            slice_inds += (kwargs[key], )
        else:
            slice_inds += (slice(0, data_dict['signal'].shape[k_ii]), )

    ### Manual procedure, no practical way of dimension checking
    # Make a DEEEEP copy of the dictionary
    new_dict = {}
    new_dict['signal'] = data_dict['signal'][slice_inds]
    new_dict['axis_ord'] = data_dict['axis_ord'][...]

    for k_ii, key in enumerate(data_dict['axis_ord']):
        new_dict[key] = {}

        for k_jj, subkey in enumerate(data_dict[key]):
            new_dict[key][subkey] = data_dict[key][subkey][slice_inds[k_ii]]

    check_dict_layout(new_dict)

    return new_dict


def combine(data_dicts, lbl):
    """
    Concatenate a list of data_dicts along an existing label dimension.

    Parameters
    ----------
        kwargs: keys correspond to axis labels, with contents of index lists or slices

    Return
    ------
        data_dict: a DEEP copy of the data_dict corresponding to the data subset.
    """

    # Check all axis ords match up
    for d in data_dicts:
        check_dict_layout(d)

    # Check label dimension is consistent
    lbl_axs = []
    for d in data_dicts:
        lbl_axs.append(get_axis(data_dicts[0], lbl))
    assert len(np.unique(lbl_axs)) == 1
    lbl_ax = lbl_axs[0]

    ### Manual procedure, no practical way of dimension checking
    # Make a DEEEEP copy of the dictionary
    new_dict = {}
    new_dict['signal'] = np.concatenate(
        [d['signal'] for d in data_dicts], axis=lbl_ax)
    new_dict['axis_ord'] = data_dicts[0]['axis_ord'][...]

    for k_ii, key in enumerate(data_dicts[0]['axis_ord']):
        new_dict[key] = {}

        if lbl != key:
            new_dict[key] = data_dicts[0][key]
        else:
            for k_jj, subkey in enumerate(data_dicts[0][key]):
                # Accumulate key data across concatenation list dicts
                subkey_arr = np.concatenate(
                    [d[key][subkey] for d in data_dicts], axis=0)
                new_dict[key][subkey] = subkey_arr

    check_dict_layout(new_dict)

    return new_dict
