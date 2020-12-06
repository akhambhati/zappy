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
from hdbscan import HDBSCAN

from zappy.elstim.waveform import locate_pulses, parse_stim_seq_to_trains


def pad_pulses(pulse_inds, padding):
    """
    Excise signal centered around each pulse.

    Parameters
    ----------
    pulse_inds: np.ndarray, shape: [2, n_pulse]
        Samples indices corresponding to the location of the pulses.

    padding: list, shape: (2,)
        Number of samples to pad around the pulses given by pulse_inds.

    Returns
    -------
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

    return pulse_inds_mod


def clip_pulses_ieeg(signal, pulse_inds):
    """
    Excise signal centered around each pulse.

    Parameters
    ----------
    signal: np.ndarray, shape: [n_sample x n_chan]
        Recorded neural signal.

    pulse_inds: np.ndarray, shape: [2, n_pulse]
        Samples indices corresponding to the location of the pulses.

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

    # Remove pulses that lay outside data range
    pulse_inds_mod = pulse_inds.copy()
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


def clip_artifact_candidates(signal, stim_seq, padding, amp_range):
    """
    Clip the window around potential stimulation artifact, termed
    "candidates" -- some stimulation pulses do not yield signal amplitude changes
    that exceed background variation. Returns a tensor of the candidates
    and their position within the original signal.

    Parameters
    ----------
    signal: np.ndarray, shape: [n_sample x n_chan]
        Recorded neural signal with stimulation pulse artifact.

    stim_seq: np.ndarray, shape: [n_sample x n_stim_chan]
        Stimulation pulse sequences over multiple electrodes.

    padding: list, shape: (2,);
        Number of samples to pad behind and ahead of isolated stim pulses.
        Increased padding helps better resolve artifact from the non-artifact
        background signal. Increasing padding too high will cause bleeding over
        to adjacent pulses in a continuous stim train.

    amp_range: tuple, shape: (float, float), default is (0, 0.009)
        The range of amplitudes (in amperes) within which to search for pulses.

    Returns
    -------
    feats_stim: np.ndarray, shape: [n_sample x n_chan x n_candidates]
        Clipped artifact candidates from the original signal.

    pulse_stim_inds: np.ndarray, shape: [2 x n_candidates]
        Onset and offset index of the candidate clips within the input signal.
    """

    # Aggregate all pulses across stim sequence
    print('Locating stimulation pulses from input sequence...')
    pulse_stim_inds = []
    for si in range(stim_seq.shape[1]):
        pinds, cinds = locate_pulses(stim_seq[:, si], amp_range=amp_range)

        # Add all pulses to aggregate list
        for pp in pinds.T:
            pulse_stim_inds.append(pp)
    pulse_stim_inds = np.array(pulse_stim_inds).T

    # Pad the pulses
    pulse_stim_inds = pad_pulses(pulse_stim_inds, padding=padding)

    # Clip the iEEG (Stim)
    print('Clipping and aggregating iEEG around stim pulses...')
    feats_stim, _ = clip_pulses_ieeg(
            signal, pulse_stim_inds)
    feats_stim = feats_stim.transpose((2,1,0))

    return feats_stim, pulse_stim_inds


def normalize_artifact_candidates(feats_stim, robust=False):
    """
    Use z-score to normalize candidates.

    Parameters
    -------
    feats_stim: np.ndarray, shape: [n_sample x n_chan x n_candidates]
        Clipped artifact candidates from the original signal.

    robust: bool, default: False
        Set to True to use robust z-score based on medians.

    Returns
    -------
    feats_stim_z: np.ndarray, shape: [n_sample x n_chan x n_candidates]
        Clipped artifact candidates from the original signal after normalization
        using a robust z-score.
    """

    # Compute the robust z-Score
    if robust:
        return (feats_stim - np.median(feats_stim, axis=0)) / \
            sp_stats.median_abs_deviation(feats_stim, axis=0)
    else:
        return sp_stats.zscore(feats_stim, axis=0)


def filter_artifact_candidates(feats_stim_z, z_threshold): 
    """
    Identify artifact candidates with large voltage deflections, termed "outliers".

    Parameters
    -------
    feats_stim_z: np.ndarray, shape: [n_sample x n_chan x n_candidates]
        Clipped and robust z-score normalized artifact candidates clips.

    z_threshold: float
        Threshold for determining whether a candidate contains outlier voltage
        deflections.

    Returns
    -------
    filter_candidates: list[list[int]], shape: [n_chan] -> [n_valid]
        Nested list of channels containing a list of valid candidate pulses.
    """

    # Iterate over channels
    n_sample, n_chan, n_cand = feats_stim_z.shape
    filter_candidates = []
    for ch in range(n_chan):
        thr_ix = np.flatnonzero((np.abs(feats_stim_z[:,0,:]) >= z_threshold).any(axis=0))
        filter_candidates.append(thr_ix)

    return filter_candidates


def hdbscan_template(
        artifact_candidates,
        valid_candidates,
        fwin,
        align_type='slope',
        align_tol=3,
        scale_type='procrustes',
        hdbscan_min_cluster_size=3,
        hdbscan_min_samples=2,
        hdbscan_nproc=1):
    """
    Unsupervised clustering of the candidate templates using hdbscan.
    This function first aligns all the candidate artifacts, extracts
    a window around the alignment center, trains an hdbscan model to learn
    artifact templates, then matches each candidate to the template dictionary.

    Parameters
    ----------
    artifact_candidates: np.ndarray, shape: [n_sample x n_candidates]
        Clipped artifact candidates from the original signal for a single channel
        (from clip_artifact_candidates).

    filter_candidates: list[int], shape: [n_valid]
        List of valid candidate pulses for a single channel
        (from filter_artifact_candidates).

    fwin: list, shape: (2,);
        Narrowest window about the alignment point that captures the most
        informative features about the artifact.

    align_type: {'slope', 'peak'}, default: slope
        Align each pulse based on the maximum change of the signal or based on
        where the peak occurs.

    align_tol: int
        Most pulses should have a similar alignment, artifact candidates
        with alignment points outside a tolerance window of
        [-align_tol, +align_tol] will be discarded from training the model.

    scale_type: {'procrustes', 'ratio'}, defaut: procrustes
        Match the artifact templates to the artifact candidates using a 
        procrustes fit or by rescaling based on the max/min ratio.


    Returns
    -------
    template_similarity: np.ndarray, shape: [n_candidates]
        The correlational similarity between an artifact candidate and the best
        matching template.

    template_assign: np.ndarray, shape: [n_candidates]
        Assignment of artifact candidates into template groups based on
        correlational similarity.

    template_scaled: np.ndarray, shape: [n_sample x n_candidates]
        Artifact templates rescaled to fit the candidate artifact. 
    """

    # Organize the outputs
    template_dict = {
        'training': {
            'valid_candidates': np.array([]),
            'hdb_model': np.array([]),
        },
        'clusters': {
            'full': np.array([]),
            'short': np.array([])
        },
        'matching': {
            'similarity': np.array([]),
            'assignment': np.array([]),
            'fitted': np.array([])
        }
    }

    # Get the dataset size
    n_sample, n_candidate = artifact_candidates.shape

    # Normalize the artifact candidates
    artifact_candidates_z = normalize_artifact_candidates(artifact_candidates)

    ### TODO: Handle this condition
    if len(valid_candidates) == 0:
        valid_candidates = [*range(n_candidate)]
    valid_candidates = np.array(valid_candidates, dtype=int)

    # Align all the artifacts according to alignment criteria
    if align_type == 'slope':
        midpt = np.argmax(np.abs(np.diff(artifact_candidates_z, axis=0)), axis=0)
    elif align_type == 'peak':
        midpt = np.argmax(np.abs(artifact_candidates_z), axis=0)
    else:
        raise NotImplemented('Alignment type does not exist.')

    ## Filter candidates based on alignment criteria
    # 1. Discard samples outside of the alignment tolerance
    valid_ix = valid_candidates.copy() #np.array([], dtype=int)
    if align_tol is not None:
        midpt_inds, midpt_cnts = np.unique(midpt, return_counts=True)
        midpt_mode = midpt_inds[np.argmax(midpt_cnts)]
        valid_ix = np.append(valid_ix,
            np.flatnonzero(
                (midpt >= (midpt_mode - align_tol)) &
                (midpt <= (midpt_mode + align_tol))))

    # 2. Discard samples outside the fwin criteria
    valid_ix = np.intersect1d(valid_ix,
            np.flatnonzero(
                ((midpt-fwin[0]) >= 0) &
                ((midpt+fwin[1]) <= n_sample)))

    # 3. Aggregate all the valid indices into one vector of candidates to use for
    # training the model.
    valid_ix = np.unique(valid_ix.flatten())
    valid_candidates = np.intersect1d(valid_candidates, valid_ix)
    midpt = midpt[valid_candidates]
    template_dict['training']['valid_candidates'] = valid_candidates

    # Get the valid z-scored artifacts
    artifact_candidates_z_valid = artifact_candidates_z[:, valid_candidates]
    n_valid = len(valid_candidates)

    if n_valid == 0:
        template_dict['matching']['similarity'] = np.zeros(n_candidate)
        template_dict['matching']['assignment'] =  np.zeros(n_candidate)
        template_dict['matching']['fitted'] = np.zeros((n_sample, n_candidate))
        return template_dict

    ## Create a training set of features by clipping the midpoint of the
    # biphasic pulse
    artifact_train = np.zeros((n_valid, np.sum(fwin)))
    for ii in range(n_valid):
        artifact_train[ii, :] = \
            artifact_candidates_z_valid[
                    (midpt[ii]-fwin[0]):(midpt[ii]+fwin[1]), ii]

    ## Train an HDBSCAN model
    print('Training HDBSCAN model of stim pulse artifacts...')
    hdbcl = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            gen_min_span_tree=True,
            core_dist_n_jobs=hdbscan_nproc,
            metric='euclidean',
            allow_single_cluster=True)
    hdbcl.fit(artifact_train)
    template_dict['training']['hdb_model'] = hdbcl

    # Get Cluster Means
    clust_train = np.array([np.mean(
        artifact_train.T[:, hdbcl.labels_ == lbl], axis=1)
        for lbl in np.unique(hdbcl.labels_)])

    clust_all = np.array([np.mean(
        artifact_candidates_z_valid[:, hdbcl.labels_ == lbl], axis=1)
        for lbl in np.unique(hdbcl.labels_)])
    print('Calculated {} Artifact Templates'.format(len(np.unique(hdbcl.labels_))))
    template_dict['clusters']['full'] = clust_all
    template_dict['clusters']['short'] = clust_train

    # Match pulses with cluster templates, rescale, and remove
    print('Matching artifact candidates with learned templates...')
    template_similarity = np.zeros(n_candidate)
    template_assign = np.zeros(n_candidate)
    template_scaled = np.zeros((n_sample, n_candidate))
    for ii in range(n_candidate):
        sel_cand = artifact_candidates[:, [ii]].copy()
        #sel_cand = sel_cand - np.median(sel_cand, axis=0)

        if scale_type == 'procrustes':
            err_fit = []
            clust_all_rescaled = []
            for tt in clust_all:
                p_res = procrustes(sel_cand.reshape(-1,1), tt.reshape(-1,1))
                err_fit.append(p_res[0])
                clust_all_rescaled.append(p_res[1])
            cl_ix = np.argmin(err_fit)
            sel_clust = clust_all_rescaled[cl_ix].flatten()

        elif scale_type == 'ratio':
            # Match using correlational similarity
            corrs = np.corrcoef(sel_cand.T, clust_all)[0, 1:]
            cl_ix = np.argmax(corrs)
            sel_clust = clust_all[cl_ix].copy()

            # Rescale based on ratio
            ratio = ((sel_cand.max() - sel_cand.min()) /
                     (sel_clust.max() - sel_clust.min()))
            sel_clust = ratio*sel_clust
        else:
            raise NotImplemented('Scaling type not implemented.')

        template_similarity[ii] = \
                sp_stats.pearsonr(sel_cand.flatten(), sel_clust.flatten())[0]
        template_assign[ii] = cl_ix
        template_scaled[:, ii] = sel_clust.flatten()
    template_dict['matching']['similarity'] = template_similarity
    template_dict['matching']['assignment'] =  template_assign
    template_dict['matching']['fitted'] = template_scaled

    return template_dict


def template_subtract(
        artifact_candidates,
        template_similarity,
        template_scaled,
        similarity_thr=0.5):
    """
    Remove artifact templates from artifact candidates on the basis of the
    similarity of their matching.

    Parameters
    ----------
    artifact_candidates: np.ndarray, shape: [n_sample x n_candidates]
        Clipped artifact candidates from the original signal for a single channel
        (from clip_artifact_candidates).

    template_similarity: np.ndarray, shape: [n_candidates]
        The correlational similarity between an artifact candidate and the best
        matching template.

    template_scaled: np.ndarray, shape: [n_sample x n_candidates]
        Artifact templates scaled to fit the candidate artifact. 

    similarity_thr: float
        Threshold of similarity above which a matching template is removed from
        the artifact candidate.

    Returns
    -------
    artifact_fixed: np.ndarray, shape: [n_sample x n_candidates]
        Artifact candidates with the matching template removed from the signal.
    """

    n_sample, n_candidate = artifact_candidates.shape

    print('Removing template artifact from artifact candidate...')
    artifact_fixed = []
    for ii in range(n_candidate):
        if template_similarity[ii] >= similarity_thr:
            artifact_fixed.append(artifact_candidates[:, ii] - template_scaled[:, ii])
        else:
            artifact_fixed.append(artifact_candidates[:, ii])
    artifact_fixed = np.array(artifact_fixed).T

    return artifact_fixed


def signal_reconstitute(signal, artifact_fixed, pulse_stim_inds, rescale_win=None):
    """
    Reconstitute the signal containing artifacts with fixed signal.

    Parameters
    ----------
    signal: np.ndarray, shape: [n_sample]
        Recorded neural signal with stimulation pulse artifact.

    artifact_fixed: np.ndarray, shape: [n_sample x n_candidates]
        Artifact candidates with the matching template removed from the signal.

    pulse_stim_inds: np.ndarray, shape: [2 x n_candidates]
        Onset and offset index of the candidate clips within the input signal.

    Returns
    -------
    signal_fixed: np.ndarray, shape: [n_sample]
        Neural signal with stimulation pulse artifact-free epochs re-inserted.
    """

    signal_fixed = signal.copy()
    for ii, (ix_on, ix_off) in enumerate(pulse_stim_inds.T):
        rescale = artifact_fixed[:, ii]
        if rescale_win is not None:
            nmax = np.max([signal_fixed[ix_on-rescale_win[0]:ix_on],
                           signal_fixed[ix_off:ix_off+rescale_win[1]]])
            nmin = np.min([signal_fixed[ix_on-rescale_win[0]:ix_on],
                           signal_fixed[ix_off:ix_off+rescale_win[1]]])

            omax = artifact_fixed[:, ii].max()
            omin = artifact_fixed[:, ii].min()

            rescale = \
                ((artifact_fixed[:, ii] - omin) / (omax-omin)) * (nmax-nmin) + nmin

            rescale = ((nmax-nmin)/(omax-omin)) * artifact_fixed[:, ii]

        signal_fixed[ix_on:ix_off] = rescale

    return signal_fixed


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform 
