from typing import Tuple, List

import numpy as np
from hmmlearn.base import BaseHMM
from hmmlearn.hmm import _check_and_set_n_features
from numpy import ndarray
from numpy.random import RandomState

offset_dim_idx = 2
offset_bins = list(range(-1, 15))


class HistogramHMM(BaseHMM):
    def __init__(self,
                 n_states: int,
                 obs_dim: int,
                 emission_hists: list,
                 emission_bins: list,
                 startprob_prior: ndarray = 1.0,
                 transmat_prior: ndarray = 1.0,
                 algorithm: str = "viterbi",
                 random_state: int = None,
                 n_iter: int = 10,
                 tol: float = 1e-2,
                 verbose: bool = False,
                 params: str = "st",
                 init_params: str = "st",
                 implementation: str = "log") -> None:

        super().__init__(n_states,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior,
                         algorithm=algorithm,
                         random_state=random_state,
                         n_iter=n_iter,
                         tol=tol,
                         params=params,
                         verbose=verbose,
                         init_params=init_params,
                         implementation=implementation)
        self.n_states = n_states
        self.obs_dim = obs_dim
        self.emission_hists = emission_hists
        self.emission_bins = emission_bins

    def _init(self, X: ndarray) -> None:
        _check_and_set_n_features(self, X)
        super()._init(X)

    def _check(self) -> None:
        super()._check()
        if len(self.emission_hists) != self.n_states:
            raise ValueError("There should be an emission histogram for each state!")
        if len(self.emission_bins) != self.n_states:
            raise ValueError("There should be a tuple of emission bin boundaries for each state!")
        for idx, emission_hist in enumerate(self.emission_hists):
            if emission_hist.ndim != self.obs_dim:
                raise ValueError("Emission histogram dimensions should match observation dimensions!")
            if np.sum(emission_hist) != 1:
                self.emission_hists[idx] = emission_hist / np.sum(emission_hist)
                print(f"Normalized histogram for state {idx}.")
        for emission_bin in self.emission_bins:
            if len(emission_bin) != self.obs_dim:
                raise ValueError("Number of emission bin boundary lists should match observation dimensions!")

    def _generate_sample_from_state(self, state: int, random_state: int = None) -> ndarray:
        # Randomly sample an index according to the histogram
        rnd = np.random.rand()
        hist = self.emission_hists[state]
        bins = self.emission_bins[state]
        cum_sum = 0
        rnd_idx = None
        for idx, prob in np.ndenumerate(hist):
            cum_sum += prob
            if cum_sum > rnd:
                rnd_idx = idx
                break
        # Uniformly sample within the histogram bin
        rnd_val = np.zeros(self.obs_dim)
        for i in range(len(rnd_idx) - 1):
            rnd_val[i] = np.random.rand() * (bins[i][rnd_idx[i] + 1] - bins[i][rnd_idx[i]]) + bins[i][rnd_idx[i]]
        rnd_val[offset_dim_idx] = offset_bins[rnd_idx[offset_dim_idx]]
        return rnd_val

    def _compute_log_likelihood(self, X):
        probs = np.zeros((X.shape[0], self.n_states))
        for sample_idx in range(X.shape[0]):
            for state_idx in range(self.n_states):
                obs_dim = X.shape[1]
                bin_idx = np.zeros(obs_dim, dtype=np.int8)
                for i in range(obs_dim):
                    try:
                        bin_idx[i] = list(np.array(self.emission_bins[state_idx][i] > X[sample_idx][i])).index(True) - 1
                    except ValueError:
                        # Histogram should include all values, but only last bin max is inclusive
                        bin_idx[i] = self.emission_bins[state_idx][i].size - 2
                if np.any(bin_idx < 0):
                    probs[sample_idx][state_idx] = 0
                else:
                    probs[sample_idx][state_idx] = self.emission_hists[state_idx][tuple(bin_idx)]
        return np.log(probs)

    def _initialize_sufficient_statistics(self) -> dict:
        stats = super()._initialize_sufficient_statistics()
        return stats

    def _accumulate_sufficient_statistics(self,
                                          stats: dict,
                                          X: ndarray,
                                          lattice: ndarray,
                                          posteriors: ndarray,
                                          fwdlattice: ndarray,
                                          bwdlattice: ndarray) -> None:
        super()._accumulate_sufficient_statistics(
            stats, X, lattice, posteriors, fwdlattice, bwdlattice)

    def _do_mstep(self, stats: dict) -> None:
        super()._do_mstep(stats)
