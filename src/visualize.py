import matplotlib.pyplot as plt
import numpy as np

from src.switch_2afc_task import STIMULI_FREQS


def get_block_indices(blocks, block_idx):
    """Return the indices of the first and last trials of a block within all trials of the experiment."""
    trial_idx = 0
    for i in range(block_idx):
        trial_idx += blocks[i][2]
    return trial_idx, trial_idx + blocks[block_idx][2]


def get_psychometric_percents(actions, rewards, stimuli, num_stimuli, action_type=None, reward_type=None,
                              stimulus_type=None):
    """Compute the percentage of left choices in a 2AFC task for different stimulus values."""
    if action_type:
        actions = [None if act != action_type else act for act in actions]
    if reward_type:
        rewards = [None if rew != reward_type else rew for rew in rewards]
    if stimulus_type:
        stimuli = [None if stim != stimulus_type else stim for stim in stimuli]

    act_rew_stim = list(zip(actions, rewards, stimuli))

    num_left_actions = np.zeros(num_stimuli)
    num_presentations = np.zeros(num_stimuli)

    for act, rew, stim in act_rew_stim:
        if act is None or rew is None or stim is None:
            continue
        elif act == 0:
            num_left_actions[stim] += 1
        num_presentations[stim] += 1

    return num_left_actions / num_presentations


def plot_psychometric_curve(block_params, rewards, stimuli, actions):
    """Plot a psychometric scatter plot."""
    block_percents = []
    fig = plt.figure()
    for block_idx in range(len(block_params)):
        psycho_rewards = rewards[block_idx]
        psycho_stimuli = stimuli[block_idx]
        action_idxs = get_block_indices(block_params, block_idx)
        psycho_actions = actions[action_idxs[0]:action_idxs[1]]
        percents = get_psychometric_percents(psycho_actions, psycho_rewards, psycho_stimuli, len(STIMULI_FREQS))
        block_percents.append(percents)
        plt.scatter(STIMULI_FREQS, percents, label=f'{block_params[block_idx][0]}')
        fig.legend()
    plt.show()
    return fig
