import numpy as np
from src.data.environments import STIMULI_IDXS


def build_observations(*args):
    """
    Takes in lists of observations and reshapes them into the correct observation format for the SSM library.
    :param args: lists of observation values of length n (corresponding to trials, blocks, etc.).
    :return: list of tuples with each type of observation of length n.
    """
    if len(set([len(arg) for arg in args])) != 1:
        print([len(arg) for arg in args])
        raise ValueError("All lists of observations should be the same length!")

    obs = list(zip(*args))
    return np.array(obs)


def switching_stimuli_psychometric(expt, match="rule"):
    """
    Compute choice correctness (incorrect: 0, correct: 1) for each trial with a presentation of a switching stimulus.
    :param expt: Experiment object with completed results.
    :param match: string indicating which definition of "correctness" to use.
    :return: list of lists of 0s and 1s with outer length equal to the number of blocks and inner lengths equal to
    the number of trials with switching stimuli in each block.
    """
    switching_trials = []
    blocks = expt.blocks
    rewards = expt.environment.reward_history
    actions = expt.agent.action_history
    stimuli = expt.environment.stimulus_idx_history
    for block in blocks:
        num_trials = block[2]
        block_switching_trials = []

        left_stimuli_idxs = STIMULI_IDXS['LEFT'].copy()
        right_stimuli_idxs = STIMULI_IDXS['RIGHT'].copy()
        boundary = block[0]
        if boundary == 'HIGH':
            left_stimuli_idxs += STIMULI_IDXS['SWITCH']
        elif boundary == 'LOW':
            right_stimuli_idxs = STIMULI_IDXS['SWITCH'] + right_stimuli_idxs

        for trial_idx in range(num_trials):
            if stimuli[trial_idx] in STIMULI_IDXS["SWITCH"]:
                # Label an action as correct if it matches the rule based on the current boundary.
                if match == "rule":
                    block_switching_trials.append(actions[trial_idx] == (stimuli[trial_idx] in right_stimuli_idxs))
                # Label an action as correct if it matches the actual rewarded action in the environment."
                elif match == "actual":
                    block_switching_trials.append(actions[trial_idx] == rewards[trial_idx])

        switching_trials.append(block_switching_trials)

    return switching_trials
