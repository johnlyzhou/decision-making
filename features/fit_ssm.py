import numpy as np
import ssm
from ssm.util import find_permutation

from src.visualization import visualize_ssm as vizssm


def run_ssm_fit(expt, obs, num_iters=1000, num_states=2, test=False):
    true_states = np.array(expt.agent.state_history)
    obs_dim = len(obs[0])
    hmm = ssm.HMM(num_states, obs_dim, observations="gaussian")
    if test:
        num_iters = 10
    hmm_lls = hmm.fit(obs, method="em", num_iters=num_iters, init_method="kmeans")

    # Permute states to most likely correct order
    most_likely_states = hmm.most_likely_states(obs)
    hmm.permute(find_permutation(true_states, most_likely_states))
    inferred_states = hmm.most_likely_states(obs)

    # Look at fitting results
    vizssm.plot_em_fitting(hmm_lls)

    # Look at inferred state results
    vizssm.plot_true_vs_inferred_states(true_states, inferred_states)

    # Look at inferred transition matrix results
    true_transition_matrix = expt.transition_matrix
    inferred_transition_matrix = hmm.transitions.transition_matrix
    vizssm.plot_true_vs_inferred_transition_matrix(true_transition_matrix, inferred_transition_matrix)