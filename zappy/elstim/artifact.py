# -*- coding: utf-8 -*-
"""
Pipelines for blanking, removing, interpolating, and reconstrucing, intracranial
EEG signal lost during stimulation pulses.

:Author: Ankit N. Khambhati
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp_sig
import scipy.stats as sp_stats
from sklearn.decomposition import FastICA

from zappy.elstim.waveform import locate_pulses, parse_stim_seq_to_trains


def clip_pulses_ieeg(signal, pulse_inds, padding=[50, 50]):
    """
    Excise signal centered around each pulse.

    Parameters
    ----------
    signal: np.ndarray, shape: [n_sample x n_chan]
        Recorded neural signal.

    pulse_inds: np.ndarray, shape: [2, n_pulse]
        Samples indices corresponding to the location of the pulses.

    padding: list, shape: (2,)
        Number of samples to pad around the pulses given by pulse_inds.

    Returns
    -------
    clipped_signal: np.ndarray,
                shape: [n_pulse, n_chan, (max_pulse_width + sum(padding))]
        Neural signal clipped around each pulse.

    pulse_inds_mod: np.ndarray, shape: [2, n_pulse]
        Samples indices corresponding to the location of the pulses, modified
        by the padding and whether the newly padded pulse indices lay outside
        the data range.
    """

    # Check pulse indices
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
                                    ~(pulse_inds_mod[1, :] > signal.shape[0])]

    n_p = pulse_inds_mod.shape[1]
    n_s = np.unique(pulse_inds_mod[1] - pulse_inds_mod[0])
    assert len(n_s) == 1
    n_s = n_s[0]
    n_c = signal.shape[1]

    clipped_signal = np.zeros((n_p, n_c, n_s))
    for ii, pinds in enumerate(pulse_inds_mod.T):
        clipped_signal[ii, :, :] = signal[pinds[0]:pinds[1], :].T

    return clipped_signal, pulse_inds_mod


def train_ica(pulse_matr, n_components=None):
    """
    Train an ICA model on matrix of concatenated pulses.

    Parameters
    ----------
    pulse_matr: np.ndarray, shape: [n_clips x n_sample]
        Matrix of excised pulses, concatenated into a two-dimensional matrix.
        First dimension corresponds to observed clips over pulses, second
        dimensions corresponds to number of signal samples per clip.

    n_components: int, default is None
        Number of ICA components to extract from the pulse matrix. 
        If None, the components will be set to the maximum possible --
        likely the number of samples per clip.

    Returns
    -------
    ica: sklearn.decomposition.FastICA object
        ICA model fitted on pulse_matr.
    """

    n_obs, n_feat = pulse_matr.shape
    ica = FastICA(n_components=n_components, max_iter=1000, tol=1e-3)
    ica = ica.fit(pulse_matr)

    return ica


def plot_ica(ica, padding=[50, 50]):
    """
    Plot trained ICA model mixing matrix.

    Parameters
    ----------
    ica: sklearn.decomposition.FastICA object
        Trained ICA model.

    padding: list, shape: (2,)
        Padding around the stim pulse used to demarcate spill-over artifact
        that exceeds the bounds of the stim pulse itself.
    """

    n_f, n_c = ica.mixing_.shape

    n_row = int(np.ceil(np.sqrt(n_c)))
    n_col = int(np.ceil(n_c / n_row))

    plt.figure(figsize=(12,12), dpi=300)
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


def reconstruct_ica(pulse_matr, ica, rm_comp=[]):
    """
    Reconstruct concatenated pulses using valid ICs.

    Parameters
    ----------
    pulse_matr: np.ndarray, shape: [n_clips x n_sample]
        Matrix of excised pulses, concatenated into a two-dimensional matrix.
        First dimension corresponds to observed clips over pulses, second
        dimensions corresponds to number of signal samples per clip.

    ica: sklearn.decomposition.FastICA object
        ICA model fitted on pulse_matr.

    rm_comp: list[int]
        List of indices corresponding to the components that should be removed
        from the model.

    Returns
    -------
    recons_pulse_matr: np.ndarray, shape: [n_clips x n_sample]
        Reconstructed matrix of excised pulses, with the components
        specified in rm_comp removed.
    """

    src = ica.transform(pulse_matr)
    src[:, rm_comp] = 0
    recons_pulse_matr = src.dot(ica.mixing_.T)

    return recons_pulse_matr


def gen_sham_inds(stim_inds, avoid_inds, array_size):
    """
    Resample the indices corresponding to stimulation in any array of length
    array_size, while avoiding any overlap with indices in avoid_inds.

    Parameters
    ----------
    stim_inds: np.ndarray, shape: [2, n_stim]
        Samples indices corresponding to the location of the stim.

    avoid_inds: np.ndarray, shape: [2, n_avoid]
        Samples indices corresponding to signal portions to avoid.

    array_size: int
        The putative length of the array from which sham indices should be
        generated.

    Parameters
    ----------
    sham_inds: np.ndarray, shape: [2, n_stim]
        Samples indices corresponding to the location of sham stim 
        (resampled indices).

    """

    # Bucket to aggregate sham indices
    sham_inds = []

    # Iterate over current stimulation indices
    for inds in stim_inds.T:
        ind_len = inds[1] - inds[0]

        # Exhaustively search for a valid index for the sham stim
        limit = 0
        while limit < 10000:

            # Randomly choose a starting index for the sham
            start_ind = np.random.randint(array_size - ind_len)
            end_ind = start_ind + ind_len

            # Check that the sham indices don't overlap with indices to avoid
            check_start = ((avoid_inds[0, :] <= start_ind) &
                           (avoid_inds[1, :] >= start_ind)).any()
            check_end = ((avoid_inds[0, :] <= end_ind) &
                         (avoid_inds[1, :] >= end_ind)).any()

            check_span = ((avoid_inds[0, :] >= start_ind) &
                          (avoid_inds[1, :] <= end_ind)).any()

            # If valid resample, then break and search for next index
            if ~(check_start | check_end | check_span):
                break

            # If not valid, reset the indices and keep searching
            start_ind = np.nan
            end_ind = np.nan
            limit += 1

        # Append the discovered sham indices
        sham_inds.append([start_ind, end_ind])

    sham_inds = np.array(sham_inds).T
    return sham_inds


def ica_pulse_reconstruction(signal,
                             stim_seq,
                             inter_train_len,
                             padding,
                             amp_range=(0, 0.009),
                             n_components=None,
                             ic_summary_stat_func=sp_stats.kurtosis,
                             ic_stat_pct=[2.5, 97.5],
                             plot=True,
                             signal_sham=None,
                             test_sham=False):
    """
    Reconstruct neural signa around individual stimulation pulses using
    ICA. This function runs ICA on the artifact signal, and compares
    the summary statistic of the uncovered sources to the summary statistic of sham signal
    of equal duration as the artifact. Components with summary statistic values outside
    the range of the sham distribution are rejected from the artifact ICA model.
    The neural signal around the pulses is reconstructed from the remaining
    components of the artifact ICA model.

    Parameters
    ----------
    signal: np.ndarray, shape: [n_sample x n_chan]
        Recorded neural signal with stimulation pulse artifact.

    stim_seq: np.ndarray, shape: [n_sample x n_stim_chan]
        Stimulation pulse sequences over multiple electrodes.

    inter_train_len: int
        Minimum number of samples that separates one stim train from the next.

    padding: list, shape: (2,);
        Number of samples to pad behind and ahead of isolated stim pulses.
        Increased padding helps better resolve artifact from the non-artifact
        background signal. Increasing padding too high will cause bleeding over
        to adjacent pulses in a continuous stim train.

    amp_range: tuple, shape: (float, float), default is (0, 0.009)
        The range of amplitudes (in amperes) within which to search for pulses.

    n_components: int, default is None
        Number of ICA components to extract from the pulse matrix. 
        If None, the components will be set to the maximum possible --
        likely the number of samples per clip.

    ic_summary_stat_func: stat function (default: scipy.stats.kurtosis)
        A function handle for computing the summary statistic of the ICs from
        stim epochs and sham epochs. Kurtosis works particularly well based on
        offline testing, as it is able to identify the peakedness of the stim
        artifact waveform.

    ic_stat_pct: list, shape: (2,)
        Percentile range of the sham summary stat distribution to use as a 
        threshold for determining artifactual components.

    plot: bool, Default is True
        Generate diagnostic plots of the summary stat distributions, ICA sources
        for the sham signal and for the artifactual signal, randomly drawn
        reconstructions around individual pulses.

    signal_sham: np.ndarray, shape: [n_sample x n_chan]
        Recorded neural signal with stimulation pulse artifact.

    test_sham: bool, Default is False
        Add artifactual signal to the sham signal, and then attempt to
        reconstruct the corrupted sham signal. Intended for testing pipeline.

    Returns
    -------
    recons_signal: np.ndarray, shape: [n_sample x n_chan]
        Neural signal with reconstructed data around the pulses.
    """

    # Signal Sham as signal
    if signal_sham is None:
        signal_sham = signal.copy()

    # Aggregate all pulses across stim sequence
    pulse_stim_inds = []
    avoid_stim_inds = []

    # Iterate over all stim sequences
    print('Locating stimulation pulses from input sequence...')
    for si in range(stim_seq.shape[1]):
        pinds, cinds = locate_pulses(stim_seq[:, si], amp_range=amp_range)

        # Add all pulses to aggregate list
        for pp in pinds.T:
            pulse_stim_inds.append(pp)

        # Use epoch booundaries as a guide for resample indices to avoid
        einds = parse_stim_seq_to_trains(stim_seq[:, si], inter_train_len,
                                         pinds)
        for pp in einds.T:
            avoid_stim_inds.append(
                [pp[0] - np.max(padding), pp[1] + np.max(padding)])

    pulse_stim_inds = np.array(pulse_stim_inds).T
    avoid_stim_inds = np.array(avoid_stim_inds).T

    # Generate a set of sham indices using aggregate set of pulse indices
    print('Determining non-pulse periods to use as control...')
    pulse_sham_inds = gen_sham_inds(pulse_stim_inds, avoid_stim_inds,
                                    signal_sham.shape[0])

    # Clip the iEEG (Stim)
    print('Clipping and aggregating iEEG around stim pulses...')
    feats_stim, pinds_stim_padded = clip_pulses_ieeg(
        signal, pulse_stim_inds, padding=padding)
    n_trial, n_chan, n_feat = feats_stim.shape
    feats_stim = feats_stim.reshape(n_trial * n_chan, n_feat)

    # Clip the iEEG (Sham)
    print('Clipping and aggregating iEEG around non-stim pulses...')
    feats_sham, pinds_sham_padded = clip_pulses_ieeg(
        signal_sham, pulse_sham_inds, padding=padding)
    n_trial1, n_chan1, n_feat1 = feats_sham.shape
    feats_sham = feats_sham.reshape(n_trial1 * n_chan1, n_feat1)

    if test_sham:
        feats_sham = feats_sham[:feats_stim.shape[0]]
        for ii in range(len(feats_stim)):
            feats_stim[ii] = (
                feats_stim[ii] * sp_sig.hanning(feats_stim.shape[1]) +
                feats_sham[ii] * (1 - sp_sig.hanning(feats_stim.shape[1])))
        pinds_stim_padded = pinds_sham_padded.copy()

    # Train ICA model
    print('Training ICA models on stim pulses and non-stim pulses...')
    ica_stim = train_ica(feats_stim, n_components=n_components)
    ica_sham = train_ica(feats_sham, n_components=n_components)

    print('Calculating summary statistic on distribution of ICs...')
    # Get summary stat of components (Stim)
    krt_stim = ic_summary_stat_func(ica_stim.mixing_, axis=0)
    ica_stim.components_ = ica_stim.components_[np.argsort(krt_stim)[::-1], :]
    ica_stim.mixing_ = ica_stim.mixing_[:, np.argsort(krt_stim)[::-1]]
    krt_stim = krt_stim[np.argsort(krt_stim)[::-1]]

    # Get summary stat of components (Sham)
    krt_sham = ic_summary_stat_func(ica_sham.mixing_, axis=0)
    ica_sham.components_ = ica_sham.components_[np.argsort(krt_sham)[::-1], :]
    ica_sham.mixing_ = ica_sham.mixing_[:, np.argsort(krt_sham)[::-1]]
    krt_sham = krt_sham[np.argsort(krt_sham)[::-1]]

    # Reconstruct components
    print('Reconstructing iEEG of stim pulses after removing corrupt components...')
    feats_stim_recons = reconstruct_ica(
        feats_stim,
        ica_stim,
        rm_comp=np.flatnonzero(
            (krt_stim < np.percentile(krt_sham, ic_stat_pct[0]))
            | (krt_stim > np.percentile(krt_sham, ic_stat_pct[1]))))

    if plot:
        # Plot Components
        print('Independent Components of Stim Pulses...')
        plot_ica(ica_stim, padding=padding)
        print('Independent Components of Non-Stim Pulses...')
        plot_ica(ica_sham, padding=padding)

        # Plot IC Summary Stat
        print('Distribution of summary stat values for IC removal...')
        plt.figure(figsize=(6,6), dpi=300)
        ax = plt.subplot(111)
        ax.plot(krt_stim)
        ax.plot(krt_sham)
        ax.hlines(np.percentile(krt_sham, ic_stat_pct),
                  0, len(krt_stim), linestyle='--', color='r')
        ax.set_xlabel('Ranked ICs')
        ax.set_ylabel('IC Summary Stat')
        ax.legend(['Stim Seq', 'Sham Seq'])
        plt.show()

        # Plot example reconstructions
        print('Example iEEG of stim pulses before/after IC removal...')
        rand_ix = np.random.permutation(len(feats_stim_recons))[:16]
        plt.figure(figsize=(6,6), dpi=300)
        for ii, ix in enumerate(rand_ix):
            ax = plt.subplot(4, 4, ii + 1)
            ax.plot(feats_stim[ix])
            ax.plot(feats_stim_recons[ix])
            ax.set_axis_off()
        plt.show()

    # Reconstitute the signal
    print('Reconstitute full iEEG with cleaned pulse periods...')
    from scipy.interpolate import interp1d
    feats_stim_recons = feats_stim_recons.reshape(n_trial, n_chan, n_feat)
    for ii, inds in enumerate(pinds_stim_padded.T):
        for jj in range(signal.shape[1]):

            # Grab the first/second-order stats of the signal around the
            # excised clip (half the padding on either side of the clip)
            pdng = inds[1]-inds[0]
            pre_ix = np.arange(inds[0]-pdng, inds[0])
            post_ix = np.arange(inds[1], inds[1] + pdng)
            all_ix = np.concatenate((pre_ix, post_ix))

            pre_sig = signal[pre_ix, jj]
            post_sig = signal[post_ix, jj]
            all_sig = signal[all_ix, jj]

            pre_rng = (pre_sig - pre_sig.mean()).max() - (pre_sig - pre_sig.mean()).min()
            post_rng = (post_sig - post_sig.mean()).max() - (post_sig - post_sig.mean()).min()

            # Linear interpolation of the mean and standard deviation across
            # the blank
            line_mean = interp1d(all_ix, all_sig)
            line_rng = np.linspace(pre_rng, post_rng, pdng)

            # Modulate the reconstructed component by the shift and scale of
            # the interpolation.
            feat_zs = feats_stim_recons[ii, jj, :].copy()

            # Detrend the reconsutrction
            slope, yint, _, _, _ = sp_stats.linregress(np.arange(len(feat_zs)), feat_zs)
            feat_zs = feat_zs - (slope*feat_zs + yint)

            # Normalize the reconstruction
            feat_zs = feat_zs - feat_zs.mean()
            feat_zs = 2*((feat_zs - feat_zs.min()) / (feat_zs.max() - feat_zs.min())) - 1

            # Rescale the reconstruction and add a trendline
            signal[inds[0]:inds[1], jj] = (feat_zs*line_rng) + line_mean(np.arange(inds[0], inds[1]))


    return signal
