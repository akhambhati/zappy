"""
Utilities to Epoch and Reconstruct iEEG around stimulation pulses.

Author: Ankit N. Khambhati
Last Updated: 2019/10/25
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
    """Construct a single stimulation pulse"""

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


def _convert_pulses_to_dirac(stim_seq, pulse_amp=0.001, ckt_open=0.01):
    """Assuming a sequence of biphasic pulses (leading up phase)"""

    stim_seq_mag = np.abs(stim_seq)
    stim_seq_ckt = stim_seq_mag.copy()
    stim_seq_cln = stim_seq_mag.copy()
    stim_seq_mag[stim_seq_mag >= ckt_open] = 0
    stim_seq_ckt[stim_seq_ckt < ckt_open] = 0
    stim_seq_cln[stim_seq_cln > 0] = 1
    stim_seq_cln = 1 - stim_seq_cln

    pulse_onset = np.flatnonzero(np.diff(stim_seq_mag) > 0) + 1
    pulse_offset = np.flatnonzero(np.diff(stim_seq_mag) < 0) + 1
    pulse_inds = np.array([pulse_onset, pulse_offset])

    ckt_opened = np.flatnonzero(np.diff(stim_seq_ckt) > 0) + 1
    ckt_closed = np.flatnonzero(np.diff(stim_seq_ckt) < 0) + 1
    ckt_inds = np.array([ckt_opened, ckt_closed])


    cln_onset = np.flatnonzero(np.diff(stim_seq_cln) < 0) + 1
    cln_offset = np.flatnonzero(np.diff(stim_seq_cln) > 0) + 1
    cln_inds = np.array([cln_onset, cln_offset])

    return pulse_inds, ckt_inds, cln_inds


def _convert_stim_seq_inst_freq(stim_seq):
    """Determine instantaneous pulse frequency from a sequence of pulses"""

    pulse_inds, _, _ = _convert_pulses_to_dirac(stim_seq)
    on_inds = pulse_inds[0, :]

    inst_freq = np.zeros_like(stim_seq)
    inst_rate = np.diff(on_inds)
    for ii in range(len(on_inds) - 1):
        inst_freq[on_inds[ii] + 1:on_inds[ii + 1]] = 1 / inst_rate[ii]

    return inst_freq


def _convert_stim_seq_to_epoch(stim_seq, max_inter_pulse):
    """Use inter-pulse widths to divide a stim seq into on/off epochs"""

    # Find stim onsets
    pulse_inds, _, _ = _convert_pulses_to_dirac(stim_seq)
    on_inds = pulse_inds[0, :]
    off_inds = pulse_inds[1, :]

    max_inter_pulse = int(np.ceil(max_inter_pulse))

    epoch = np.ones(len(stim_seq))
    epoch[:on_inds[0]] = 0
    for ii, (ix_1, ix_2) in enumerate(zip(on_inds[:-1], on_inds[1:])):
        if (ix_2 - ix_1) > max_inter_pulse:
            epoch[off_inds[:-1][ii]:ix_2] = 0
    epoch[(off_inds[-1] + max_inter_pulse):] = 0

    epoch_onset = np.flatnonzero(np.diff(epoch) > 0) + 1
    epoch_offset = np.flatnonzero(np.diff(epoch) < 0) + 1
    epoch_inds = np.array([epoch_onset, epoch_offset])

    return epoch_inds


def _clip_pulses_ieeg(df_ieeg, pulse_inds, padding=[50, 50]):
    """Excise iEEG centered around each pulse"""

    ### Check pulse indices
    pulse_dur = np.unique(pulse_inds[1] - pulse_inds[0])
    if len(pulse_dur) > 1:
        print('Warning: Pulses of different durations detected.' +
              'Clipping based on maximum pulse duration.')
    pulse_dur = np.max(pulse_dur)

    # Modify pulse range based on max duration and padding factor
    pulse_inds_mod = pulse_inds.copy()
    pulse_inds_mod[1, :] = pulse_inds_mod[0, :] + pulse_dur
    pulse_inds_mod[0, :] = pulse_inds_mod[0, :] - padding[0]
    pulse_inds_mod[1, :] = pulse_inds_mod[1, :] + padding[1]

    # Get unique on/off index pairs
    pulse_inds_mod = np.unique(pulse_inds_mod, axis=1)

    # Remove pulses that lay outside data range
    pulse_inds_mod = pulse_inds_mod[:, ~(pulse_inds_mod[0, :] < 0)]
    pulse_inds_mod = pulse_inds_mod[:,
                                    ~(pulse_inds_mod[1, :] > df_ieeg.shape[0])]

    n_p = pulse_inds_mod.shape[1]
    n_s = np.unique(pulse_inds_mod[1] - pulse_inds_mod[0])
    assert len(n_s) == 1
    n_s = n_s[0]
    n_c = df_ieeg.shape[1]

    clipped_pulse = np.zeros((n_p, n_c, n_s))
    for ii, pinds in enumerate(pulse_inds_mod.T):
        clipped_pulse[ii, :, :] = df_ieeg[pinds[0]:pinds[1], :].T

    return clipped_pulse, pulse_inds_mod


def _train_ica(pulse_matr, n_components=None):
    """Train an ICA model on matrix of concatenated pulses"""

    n_obs, n_feat = pulse_matr.shape
    ica = FastICA(n_components=n_components, max_iter=1000)
    ica = ica.fit(pulse_matr)

    return ica


def _plot_ica(ica, padding=[50, 50]):
    """Plot trained ICA model mixing matrix"""

    n_f, n_c = ica.mixing_.shape

    n_row = int(np.ceil(np.sqrt(n_c)))
    n_col = int(np.ceil(n_c / n_row))

    plt.figure()
    for ii in range(n_c):
        ax = plt.subplot(n_row, n_col, ii + 1)
        ax.plot(ica.mixing_[:, ii], linewidth=0.5, color='k')
        ax.vlines(
            padding[0],
            ax.get_ylim()[0],
            ax.get_ylim()[1],
            color='r',
            linewidth=0.25)
        ax.vlines(
            n_f - padding[1],
            ax.get_ylim()[0],
            ax.get_ylim()[1],
            color='r',
            linewidth=0.25)
        ax.set_title('Comp.: {}'.format(ii))
        ax.set_axis_off()
    plt.show()


def _reconstruct_ica(pulse_matr, ica, rm_comp=[]):
    """Reconstruct concatenated pulses using valid ICs"""

    src = ica.transform(pulse_matr)
    src[:, rm_comp] = 0
    recons = src.dot(ica.mixing_.T)

    return recons


def _gen_sham_inds(stim_inds, avoid_inds, array_size):

    sham_inds = []
    for inds in stim_inds.T:
        ind_len = inds[1] - inds[0]

        limit = 0
        while limit < 10000:
            start_ind = np.random.randint(array_size - ind_len)
            end_ind = start_ind + ind_len

            check_start = ((avoid_inds[0, :] <= start_ind) &
                           (avoid_inds[1, :] >= start_ind)).any()
            check_end = ((avoid_inds[0, :] <= end_ind) &
                         (avoid_inds[1, :] >= end_ind)).any()

            check_span = ((avoid_inds[0, :] >= start_ind) &
                          (avoid_inds[1, :] <= end_ind)).any()

            if ~(check_start | check_end | check_span):
                break

            start_ind = np.nan
            end_ind = np.nan
            limit += 1

        sham_inds.append([start_ind, end_ind])

    sham_inds = np.array(sham_inds).T
    return sham_inds


def clean_pulse_artifact():
    return None


def clean_ckt_artifact():
    return None


def run_ica(iEEG,
            stim_seq,
            n_components=None,
            padding=[50, 50],
            krt_pct=[10, 90],
            clean_ckt=False,
            plot=True,
            test_sham=False):
    """ICA Pipeline"""

    ### Aggregate all pulses across stim sequence
    pulse_stim_inds = []
    ckton_stim_inds = []
    cktoff_stim_inds = []
    avoid_stim_inds = []
    for si in range(stim_seq.shape[1]):
        pinds, ckinds, _ = _convert_pulses_to_dirac(stim_seq[:, si])
        for pp in pinds.T:
            pulse_stim_inds.append(pp)
        for pp in ckinds[0]:
            ckton_stim_inds.append([pp - padding[0], pp + padding[1]])
        for pp in ckinds[1]:
            cktoff_stim_inds.append([pp - padding[0], pp + padding[1]])
        einds = _convert_stim_seq_to_epoch(stim_seq[:, si], 24414)
        for pp in einds.T:
            avoid_stim_inds.append([pp[0] - np.max(padding), pp[1]+np.max(padding)])
    pulse_stim_inds = np.array(pulse_stim_inds).T
    ckton_stim_inds = np.array(ckton_stim_inds).T
    cktoff_stim_inds = np.array(cktoff_stim_inds).T
    avoid_stim_inds = np.array(avoid_stim_inds).T

    pulse_sham_inds = _gen_sham_inds(pulse_stim_inds, avoid_stim_inds,
                                     iEEG.shape[0])
    ckton_sham_inds = _gen_sham_inds(ckton_stim_inds, avoid_stim_inds,
                                     iEEG.shape[0])
    cktoff_sham_inds = _gen_sham_inds(cktoff_stim_inds, avoid_stim_inds,
                                      iEEG.shape[0])

    new_padding = padding.copy()
    if clean_ckt:
        pulse_stim_inds = cktoff_stim_inds.copy()
        pulse_sham_inds = cktoff_sham_inds.copy()
        new_padding = [0, 0]

    ### Clip the iEEG (Stim)
    feats_stim, pinds_stim_padded = _clip_pulses_ieeg(
        iEEG, pulse_stim_inds, padding=new_padding)
    n_trial, n_chan, n_feat = feats_stim.shape
    feats_stim = feats_stim.reshape(n_trial * n_chan, n_feat)

    ### Clip the iEEG (Sham)
    feats_sham, pinds_sham_padded = _clip_pulses_ieeg(
        iEEG, pulse_sham_inds, padding=new_padding)
    n_trial1, n_chan1, n_feat1 = feats_sham.shape
    feats_sham = feats_sham.reshape(n_trial1 * n_chan1, n_feat1)

    if test_sham:
        feats_sham = feats_sham[:feats_stim.shape[0]]
        for ii in range(len(feats_stim)):
            feats_stim[ii] = (
                feats_stim[ii] * sp_sig.hanning(feats_stim.shape[1]) +
                feats_sham[ii] * (1 - sp_sig.hanning(feats_stim.shape[1])))
        pinds_stim_padded = pinds_sham_padded.copy()

    if clean_ckt is False:
        ### Train ICA model
        ica_stim = _train_ica(feats_stim, n_components=n_components)
        ica_sham = _train_ica(feats_sham, n_components=n_components)

        ### Get Kurtosis of components (Stim)
        krt_stim = sp_stats.kurtosis(ica_stim.mixing_, axis=0, fisher=True)
        ica_stim.components_ = ica_stim.components_[np.argsort(krt_stim)[::-1], :]
        ica_stim.mixing_ = ica_stim.mixing_[:, np.argsort(krt_stim)[::-1]]
        krt_stim = krt_stim[np.argsort(krt_stim)[::-1]]

        ### Get Kurtosis of components (Sham)
        krt_sham = sp_stats.kurtosis(ica_sham.mixing_, axis=0, fisher=True)
        ica_sham.components_ = ica_sham.components_[np.argsort(krt_sham)[::-1], :]
        ica_sham.mixing_ = ica_sham.mixing_[:, np.argsort(krt_sham)[::-1]]
        krt_sham = krt_sham[np.argsort(krt_sham)[::-1]]

        ### Reconstruct components
        feats_stim_recons = _reconstruct_ica(
            feats_stim,
            ica_stim,
            rm_comp=np.flatnonzero(
                (krt_stim < np.percentile(krt_sham, krt_pct[0]))
                | (krt_stim > np.percentile(krt_sham, krt_pct[1]))))

        if plot:
            ### Plot Components
            if plot:
                _plot_ica(ica_stim, padding=padding)
                _plot_ica(ica_sham, padding=padding)

            ### Plot IC Kurtosis
            plt.figure()
            ax = plt.subplot(111)
            ax.plot(krt_stim)
            ax.plot(krt_sham)
            ax.set_xlabel('Ranked ICs')
            ax.set_ylabel('IC Kurtosis')
            ax.legend(['Stim Seq', 'Sham Seq'])
            plt.show()

    else:
        feats_stim_recons = feats_stim.copy()
        feats_stim_recons -= feats_stim_recons.mean(axis=0)

    if plot:
        ### Plot example reconstructions
        rand_ix = np.random.permutation(len(feats_stim_recons))[:16]
        plt.figure()
        for ii, ix in enumerate(rand_ix):
            ax = plt.subplot(4, 4, ii + 1)
            ax.plot(feats_stim[ix])
            ax.plot(feats_stim_recons[ix])
            ax.set_axis_off()

    ### Reconstitute the signal
    feats_stim_recons = feats_stim_recons.reshape(n_trial, n_chan, n_feat)
    for ii, inds in enumerate(pinds_stim_padded.T):
        for jj in range(iEEG.shape[1]):
            pre_mean = iEEG[(inds[0] - padding[0] // 2):inds[0], jj].mean()
            post_mean = iEEG[inds[1]:(inds[1] + padding[0] // 2), jj].mean()
            pre_std = iEEG[(inds[0] - padding[0] // 2):inds[0], jj].std()
            post_std = iEEG[inds[1]:(inds[1] + padding[1] // 2), jj].std()

            line_mean = np.linspace(pre_mean, post_mean, inds[1] - inds[0])
            line_std = np.linspace(pre_std, post_std, inds[1] - inds[0])

            #feat_zs = feats_stim_recons[ii,jj,:] - feats_stim_recons[ii,jj,:].mean()
            #iEEG[inds[0]:inds[1], jj] = (feat_zs + line_mean)

            feat_zs = sp_stats.zscore(feats_stim_recons[ii, jj, :])
            iEEG[inds[0]:inds[1], jj] = (feat_zs * line_std) + line_mean

    return iEEG
