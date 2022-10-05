import argparse

import numpy as np
import ssm
from ssm.util import find_permutation

from src.data.environments import STIMULI_FREQS, BOUNDARY_FREQS, BOUNDARY_IDX
from src.data.experiments import Experiment
from src.features.build_ssm_features import build_outputs
import src.visualization.visualize_ssm as vizssm
from src.utils import blockify


def run(config_path, num_iters=1000, num_states=2, obs_type="trials", test=False):
    # Run a simulated experiment according to the config file settings
    expt = Experiment(config_path)
    expt.run()
    # Get a few attributes for visualization and analysis
    rewards = expt.environment.reward_history
    stimuli = [STIMULI_FREQS[stim_idx] for stim_idx in expt.environment.stimulus_idx_history]
    actions = expt.agent.action_history
    boundaries = [BOUNDARY_FREQS[BOUNDARY_IDX[bound_key]] for bound_key in expt.environment.boundary_history]
    observations = build_outputs(actions, rewards)
    true_states = np.array(expt.agent.state_history)

    # Fit on a block-by-block basis instead of trial-by-trial
    # observations = blockify(expt.blocks, observations)
    # print(len(observations))
    # print(len(observations[0]))

    # Check psychometric plots of block transitions to make sure they make sense
    if not test:
        expt.plot_psychometric_scatter()

    # Fit HMM to observed data
    obs_dim = len(observations[0])
    hmm = ssm.HMM(num_states, obs_dim, observations="gaussian")
    if test:
        num_iters = 10
    hmm_lls = hmm.fit(observations, method="em", num_iters=num_iters, init_method="kmeans")

    # Permute states to most likely correct order
    most_likely_states = hmm.most_likely_states(observations)
    hmm.permute(find_permutation(true_states, most_likely_states))
    inferred_states = hmm.most_likely_states(observations)

    # Look at fitting results
    if not test:
        vizssm.plot_em_fitting(hmm_lls)

    # Look at inferred state results
    if not test:
        vizssm.plot_true_vs_inferred_states(true_states, inferred_states)

    # Look at inferred transition matrix results
    true_transition_matrix = expt.transition_matrix
    inferred_transition_matrix = hmm.transitions.transition_matrix
    if not test:
        vizssm.plot_true_vs_inferred_transition_matrix(true_transition_matrix, inferred_transition_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a simulated 2AFC task and fit a state space model.')
    parser.add_argument('config', help='A required path to experiment configuration file.')
    parser.add_argument('--test', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    run(args.config, test=args.test)
