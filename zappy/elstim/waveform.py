# -*- coding: utf-8 -*-
"""
Utilities to construct and deconstruct stimulation waveforms into fundamental
parameters.

:Author: Ankit N. Khambhati
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.signal as sp_sig
import scipy.stats as sp_stats
from sklearn.decomposition import FastICA


def construct_stim_pulse(n_down_phase, n_up_phase):
    """
    Construct a single stimulation pulse with different length phases.
    Currently assumes waveform is of rectangular shape.

    Parameters
    ----------
    n_down_phase: int
        Number of samples for the down phase.

    n_up_phase: int
        Number of samples for the up phase.

    Returns
    -------
    pulse: np.ndarray, shape: (n_down_phase + n_up_phase,)
        Single pulse waveform, with no DC shift.
    """

    if (n_down_phase < 0) & (n_up_phase < 0):
        raise Exception(
            'Down phase and up phase are both shorter than 1 sample')
    elif (n_down_phase < 0) & (n_up_phase > 0):
        return np.ones(n_up_phase)
    elif (n_down_phase > 0) & (n_up_phase < 0):
        return -np.ones(n_down_phase)
    elif (n_down_phase > 0) & (n_up_phase > 0):
        return np.concatenate((-np.ones(n_down_phase), np.ones(n_up_phase)))


def construct_stim_seq(pulse, n_inter_pulse=[]):
    """
    Construct a sequence of stim pulses with a pre-defined number of samples
    intervening between consecutive pulses.

    Parameters
    ----------
    pulse: np.ndarray, shape: (n_sample,)
        A waveform for a single pulse.

    n_inter_pulse: np.ndarray, shape: (n_pulse,)
        An array of samples to append between pulses. Inter-pulse period is
        assumed to have no DC shift (equal to zero).

    Returns
    -------
    pulse_seq: np.ndarray, shape: (n_sample*n_pulse + sum(n_inter_pulse),)
        An array for a complete sequence of pulses, where each pulse is
        identically parameterized based on `pulse`.
    """

    # Inter-stim counter begins at the onset of the first phase of the pulse
    # Samples during the pulse are considered a part of the inter-stim duration

    pulse_seq = np.array([])
    for n_s in n_inter_pulse:
        pulse_seq = np.concatenate((pulse_seq, pulse,
                                    np.zeros(n_s - len(pulse))))

    return pulse_seq


def locate_pulses(stim_seq, amp_range=(0, 0.001)):
    """
    Given a sequence of pulses, find the indices corresponding to
    onset and offset times. Function assumes biphasic pulses, where
    onset refers to rise of the first pulse phase and offset refers
    to fall of the second pulse phase.

    Parameters
    ----------
    stim_seq: np.ndarray, shape: [n_sample,]
        A sequence of pulses, where amplitude is assumed to be in
        units of `amperes`.

    amp_range: tuple, shape: (float, float), default is (0, 0.001)
        The range of amplitudes (in amperes) within which to search for pulses.

    Returns
    -------
    pulse_inds: np.ndarray, shape: [n_pulse, 2]
        Samples indices corresponding to the location of the pulses.

    clean_inds: np.ndarray, shape: [n_pulse+1, 2]
        Samples indices corresponding to segments without any pulses,
        regardless of amp_range criteria (portions where
        magnitude of stim_seq = 0).
    """

    # Retrieve magnitude of the sitmulation
    stim_seq_mag = np.abs(stim_seq)
    stim_seq_cln = stim_seq_mag.copy()

    # Filter the range of magnitudes based on amp_range
    stim_seq_mag[stim_seq_mag < amp_range[0]] = 0
    stim_seq_mag[stim_seq_mag > amp_range[1]] = 0

    # Also retrieve the segments where there are no pulses
    stim_seq_cln[stim_seq_cln > 0] = 1
    stim_seq_cln = 1 - stim_seq_cln

    pulse_onset = np.flatnonzero(np.diff(stim_seq_mag) > 0)
    pulse_offset = np.flatnonzero(np.diff(stim_seq_mag) < 0) + 1
    pulse_inds = np.array([pulse_onset, pulse_offset])

    clean_onset = np.flatnonzero(np.diff(stim_seq_cln) < 0)
    clean_offset = np.flatnonzero(np.diff(stim_seq_cln) > 0) + 1

    if (len(clean_offset) == 0) | (len(clean_onset) == 0):
        clean_offset = []
        clean_onset = []
    else:

        # Flip the clean onset/offset depending on the edge case
        if clean_offset[0] < clean_onset[0]:
            clean_tmp = clean_offset.copy()
            clean_offset = clean_onset.copy()
            clean_onset = clean_tmp.copy()

    # Ensure there are an equal number of onsets and offsets
    clip = np.min([len(clean_onset), len(clean_offset)])

    clean_inds = np.array([clean_onset[:clip], clean_offset[:clip]])

    return pulse_inds, clean_inds


def parse_stim_seq_to_trains(stim_seq, max_inter_pulse, pulse_inds):
    """
    Parse a sequence of stimulation pulses of varying inter_pulse intervals,
    into discrete trains (also known as trials).

    Parameters
    ----------
    stim_seq: np.ndarray, shape: [n_sample,]
        A sequence of pulses, where amplitude is assumed to be in
        units of `amperes`.

    max_inter_pulse: int
        Maximum interval between pulses (in samples) such that the sequence of
        pulses can be considered a train.

    pulse_inds: np.ndarray, shape: [n_pulse, 2]
        Samples indices corresponding to the location of the pulses.

    Returns
    -------
    epoch_inds: np.ndarray, shape: [n_epoch, 2]
        The onset and offset indices within which the inter-pulse interval
        remains below the max_inter_pulse parameter.
    """

    # Find pulse onsets and offsets
    on_inds = pulse_inds[0, :]
    off_inds = pulse_inds[1, :]

    if not ((len(on_inds) == 0) | (len(off_inds) == 0)):
        # Ensure the max inter-pulse interval is an integer
        max_inter_pulse = int(np.ceil(max_inter_pulse))

        # Construct an indicator function for epochs
        epoch = np.ones(len(stim_seq))
        epoch[:on_inds[0]] = 0
        for ii, (ix_1, ix_2) in enumerate(zip(on_inds[:-1], on_inds[1:])):
            if (ix_2 - ix_1) > max_inter_pulse:
                epoch[off_inds[:-1][ii]:ix_2] = 0
        epoch[off_inds[-1]:] = 0

        # Use the boundaries of the indicator function to demarcate the
        # onset and offset samples corresponding to an epoch.
        epoch_onset = np.flatnonzero(np.diff(epoch) > 0) + 1
        epoch_offset = np.flatnonzero(np.diff(epoch) < 0) + 1
    else:
        epoch_onset = []
        epoch_offset = []
    epoch_inds = np.array([epoch_onset, epoch_offset])

    return epoch_inds


def calc_instantaneous_frequency(stim_seq, pulse_inds):
    """
    Use the stimulation sequence to infer an instaneous pulse frequency signal.

    Parameters
    ----------
    stim_seq: np.ndarray, shape: [n_sample,]
        A sequence of pulses, where amplitude is assumed to be in
        units of `amperes`.

    pulse_inds: np.ndarray, shape: [n_pulse, 2]
        Samples indices corresponding to the location of the pulses.

    Returns
    -------
    inst_freq: np.ndarray, shape: [n_sample,]
        A continuous signal corresponding to the instantaneous frequency
        of the stimulation pulses, in unit of cycles/sample.
    """

    # Get pulse onsets and offsets
    on_inds = pulse_inds[0, :]
    off_inds = pulse_inds[1, :]

    # Compute the instaneous frequency for this epoch
    inst_freq = np.zeros_like(stim_seq)
    inst_rate = np.diff(on_inds)
    for p_ii in range(len(on_inds) - 1):
        inst_freq[on_inds[p_ii] + 1:on_inds[p_ii + 1]] = 1 / inst_rate[p_ii]

    return inst_freq


def get_stim_param_per_epoch(stim_seq, pulse_inds, epoch_inds):
    """
    Use the stimulation sequence to infer parameters regarding
    the stimulation waveform within epochs.

    Parameters
    ----------
    stim_seq: np.ndarray, shape: [n_sample,]
        A sequence of pulses, where amplitude is assumed to be in
        units of `amperes`.

    pulse_inds: np.ndarray, shape: [n_pulse, 2]
        Samples indices corresponding to the location of the pulses.

    epoch_inds: np.ndarray, shape: [n_epoch, 2]
        The onset and offset indices within which the pulse sequence
        can be considered a single `train` or an epoch.

    Returns
    -------
    epoch_params: dict of lists
        `mean_pulse_frequency` - Average frequency of pulses per epoch
        `rms_pulse_amplitude` - Root-mean-square amplitude of pulses per epoch
        `mean_pulse_amplitude` - Average amplitude of the pulses per epoch
        `mean_pulse_width` - Average length of the half-pulse width per epoch
        All time related parameters are in units of samples, and need to be
        converted to units of time.
    """

    epoch_params = {
        'mean_pulse_frequency': [],
        'rms_pulse_amplitude': [],
        'mean_pulse_amplitude': [],
        'mean_pulse_width': [],
        'mean_epoch_duration': []
    }

    for ep_ii, (ep_on, ep_off) in enumerate(epoch_inds.T):

        # Grab the sequence for this epoch
        sel_seq = stim_seq[ep_on:ep_off]

        # Select pulse inds that occur within this epoch
        p_bool = ((pulse_inds[0, :] >= ep_on) & (pulse_inds[1, :] <= ep_off))
        p_inds = pulse_inds[:, p_bool]

        # Calculate mean frequency
        epoch_params['mean_pulse_frequency'].append(
            1 / np.diff(p_inds[0]).mean())

        # Calculate rms pulse amplitude
        epoch_params['rms_pulse_amplitude'].append(
            np.sqrt(np.mean(sel_seq**2)))

        # Calculate rms pulse amplitude
        sel_seq_mag = np.abs(sel_seq)
        epoch_params['mean_pulse_amplitude'].append(
            (sel_seq_mag[sel_seq_mag > 0]).mean())

        # Calculate mean pulse width
        epoch_params['mean_pulse_width'].append(
            (0.5 * (p_inds[1] - p_inds[0])).mean())

        # Calculate mean epoch duration
        epoch_params['mean_epoch_duration'].append(ep_off - ep_on)

    # Convert lists to numpy arrays for ease of manipulation
    for key in epoch_params:
        epoch_params[key] = np.array(epoch_params[key])

    return epoch_params
