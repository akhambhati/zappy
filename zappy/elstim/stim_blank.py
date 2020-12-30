"""
Utilities to blank stimulation pulses.

Author: Ankit N. Khambhati
Last Updated: 2018/10/23
"""

import numpy as np
import scipy.stats as stats


def construct_stim_pulse(n_down_phase, n_up_phase):
    """Construct a single stimulation pulse"""

    if (n_down_phase < 0) & (n_up_phase < 0):
        raise Exception(
            'Down phase and up phase are both shorter than 1 sample')
    elif (n_down_phase < 0) & (n_up_phase > 0):
        return np.ones(n_up_phase)
    elif (n_down_phase > 0) & (n_up_phase < 0):
        return -np.ones(n_down_phase)
    elif (n_down_phase > 0) & (n_up_phase > 0):
        return np.concatenate((-np.ones(n_down_phase), [0],
                               np.ones(n_up_phase)))


def construct_stim_seq(pulse, n_inter_pulse=[]):
    """Construct one sequence of stim pulses"""

    # Inter-stim counter begins at the onset of the first phase of the pulse
    # Samples during the pulse are considered a part of the inter-stim duration

    dirac = np.array([])
    seq = np.array([])
    for n_s in n_inter_pulse:
        seq = np.concatenate((seq, pulse, np.zeros(n_s - len(pulse))))
        dirac = np.concatenate((dirac, np.ones(len(pulse)),
                                np.zeros(n_s - len(pulse))))

    return seq, dirac


def convert_seq_to_dirac(stim_seq):
    """Assuming a sequence of biphasic pulses (leading down phase)"""

    onset = np.flatnonzero(np.diff(stim_seq) < 0)[0::2] + 1
    offset = np.flatnonzero(np.diff(stim_seq) < 0)[1::2] + 1

    dirac = np.zeros(len(stim_seq))
    for ix_1, ix_2 in zip(onset, offset):
        dirac[ix_1:ix_2] = 1

    return dirac


def epoch_stim_on_off(stim_dirac, n_inter_pulse=[]):
    """Use inter-pulse widths to divide a stim dirac into on/off epochs"""

    if not stim_dirac.any():
        return np.zeros(len(stim_dirac))

    # Find stim onsets
    onset = np.flatnonzero(np.diff(stim_dirac) > 0) + 1

    # Use max inter_stim period as a threshold for epoching
    max_inter = np.max(n_inter_pulse)

    epoch = np.ones(len(stim_dirac))
    epoch[:onset[0]] = 0
    for ii, (ix_1, ix_2) in enumerate(zip(onset[:-1], onset[1:])):
        if (ix_2 - ix_1) > max_inter:
            epoch[(ix_1 + max_inter):ix_2] = 0
    epoch[(onset[-1] + max_inter):] = 0

    return epoch


def corrupt_stim_sample(stim_dirac, n_signal_len, n_pre_pad=0, n_post_pad=0):
    """Identify corrupt signal samples based on the stim dirac indicator"""

    # Tile the stim_dirac sequence to the length of the signal
    n_dirac_per_signal = n_signal_len / len(stim_dirac)
    n_dirac_per_signal = int(np.ceil(n_dirac_per_signal))
    stim_dirac_tile = np.tile(stim_dirac, n_dirac_per_signal)

    # Truncate any excess stim samples to the length of the recorded signal
    stim_dirac_tile = stim_dirac_tile[:n_signal_len]

    # Use the padding to extend the dirac boundaries
    # (samples which would be corrupt due to artifact related to stim)
    onset = np.flatnonzero(np.diff(stim_dirac_tile) == 1)
    for ix in onset:
        stim_dirac_tile[(ix - n_pre_pad + 1):(ix + 1)] = 1

    offset = np.flatnonzero(np.diff(stim_dirac_tile) == -1)
    for ix in offset:
        stim_dirac_tile[(ix + 1):(ix + n_post_pad + 1)] = 1

    return stim_dirac_tile


def blanks_as_inds(blank):
    """Identify indices that contain stimulation artifact that need blanking"""

    # Get the blank cutoffs
    blnk_on = np.flatnonzero(np.diff(blank) > 0) + 1
    blnk_off = np.flatnonzero(np.diff(blank) < 0) + 1

    # Discard blanks that are trailing at the head or tail of the vector
    if (blnk_off[0] < blnk_on[0]):
        blnk_off = blnk_off[1:]

    if (blnk_on[-1] > blnk_off[-1]):
        blnk_on = blnk_on[:-1]

    assert len(blnk_off) == len(blnk_on)

    return blnk_on, blnk_off


def frequency_notch_list(line_freq=60, n_inter_pulse=[], Fs=None):
    """Return list of corrupted frequency bands"""

    ## Get harmonics of line and stim frequencies
    line_harm = line_freq * np.arange(1, (Fs / line_freq))
    stim_freq = Fs / np.array(n_inter_pulse)

    # Concatenate all combinations of line and stim frequencies
    noise_harm = []
    for f1 in line_harm:
        noise_harm.append(f1)
        for f2 in stim_freq:
            noise_harm.append(f2)
            noise_harm.append(f2 + f1)
            noise_harm.append(f2 - f1)
    noise_harm = np.array(noise_harm)

    noise_harm = np.unique(np.abs(noise_harm))
    noise_harm = noise_harm[(noise_harm > 0) & (noise_harm < (Fs / 2))]

    return noise_harm


def linterp_blank(signal, blank):
    """Linear interpolation over blanks"""

    if not blank.any():
        return signal

    # Get the blank cutoffs
    blnk_on = np.flatnonzero(np.diff(blank) > 0) + 1
    blnk_off = np.flatnonzero(np.diff(blank) < 0)

    # Discard blanks that are trailing at the head or tail of the vector
    if (blnk_off[0] < blnk_on[0]):
        blnk_off = blnk_off[1:]

    if (blnk_on[-1] > blnk_off[-1]):
        blnk_on = blnk_on[:-1]

    assert len(blnk_off) == len(blnk_on)

    # Iterate over blanks and linearly interpolate
    for on, off in zip(blnk_on, blnk_off):
        m, yint, _, _, _ = stats.linregress([on - 1, off + 1],
                                            [signal[on - 1], signal[off + 1]])
        signal[on:off] = m * np.arange(on, off) + yint

    return signal


def revinterp_blank(signal, blank):
    """Signal reversal interpolation over blanks"""

    if not blank.any():
        return signal

    # Get the blank cutoffs
    blnk_on = np.flatnonzero(np.diff(blank) > 0) + 1
    blnk_off = np.flatnonzero(np.diff(blank) < 0)

    # Discard blanks that are trailing at the head or tail of the vector
    if (blnk_off[0] < blnk_on[0]):
        blnk_off = blnk_off[1:]

    if (blnk_on[-1] > blnk_off[-1]):
        blnk_on = blnk_on[:-1]

    assert len(blnk_off) == len(blnk_on)

    # Iterate over blanks and linearly interpolate
    for on, off in zip(blnk_on, blnk_off):         
        try:
            dur = off-on
            wt = np.linspace(0, 1, dur)
            prev = signal[on-dur:on][::-1]
            fwd = signal[off:off+dur][::-1]
            signal[on:off] = prev*wt[::-1] + fwd*wt
        except:
            m, yint, _, _, _ = stats.linregress([on - 1, off + 1],
                                                [signal[on - 1], signal[off + 1]])
            signal[on:off] = m * np.arange(on, off) + yint

    return signal


def preproc_pipeline(data_dict,
                     ds_fac=15,
                     n_inter_pulse=[244],
                     n_pre_pad=5,
                     n_post_pad=65,
                     dirac_stim_on=None):

    ### Handle the Dirac formulation
    # Get the dirac stim representation
    dirac = convert_seq_to_dirac(data_dict['sample']['stim'])

    # Epoch the stim data
    epoch = epoch_stim_on_off(dirac, n_inter_pulse=n_inter_pulse)

    if epoch.any():
        # Grab the dirac representation during a stim-ON epoch
        on_start = np.flatnonzero(np.diff(epoch) == 1) + 1
        on_end = np.flatnonzero(np.diff(epoch) == -1) + 1
        dirac_stim_on = dirac[on_start[0]:on_end[0]]

        # Iterate over each inter epoch (on_end marks start, on_start marks end)
        on_end = np.concatenate(([0], on_end))
        on_start = np.concatenate((on_start, [len(epoch)]))
        for ep_ii, (ep_start, ep_end) in enumerate(zip(on_end, on_start)):
            dirac[ep_start:ep_end] = corrupt_stim_sample(
                dirac_stim_on, ep_end - ep_start)
    else:
        # Assume whole file is inter epoch, so use n_inter_pulse to
        # construct the dirac and stim blanks
        if dirac_stim_on is None:
            raise Exception(
                'Need to input a dirac_stim_on vector, no stim sequence in supplied data_dict'
            )
        on_end = [0]
        on_start = [len(epoch)]
        for ep_ii, (ep_start, ep_end) in enumerate(zip(on_end, on_start)):
            dirac[ep_start:ep_end] = corrupt_stim_sample(
                dirac_stim_on, ep_end - ep_start)

    # Get the dirac to fit the actual signal, and extend corrupt window
    blank = corrupt_stim_sample(
        dirac,
        len(data_dict['sample']['timestamp']),
        n_pre_pad=n_pre_pad,
        n_post_pad=n_post_pad)

    # Add dirac and epoch marker to the data_dict
    data_dict['sample']['blank'] = blank.copy()
    data_dict['sample']['epoch'] = epoch.copy()

