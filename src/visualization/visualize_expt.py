import matplotlib.pyplot as plt
import numpy as np

from src.data.environments import STIMULI_FREQS
from src.utils import get_block_indices


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
        block_start, block_end = get_block_indices(block_params)[block_idx]
        psycho_rewards = rewards[block_start:block_end]
        psycho_stimuli = stimuli[block_start:block_end]
        psycho_actions = actions[block_start:block_end]
        percents = get_psychometric_percents(psycho_actions, psycho_rewards, psycho_stimuli, len(STIMULI_FREQS))
        block_percents.append(percents)
        plt.scatter(STIMULI_FREQS, percents, label=f'{block_params[block_idx][0]}')
        fig.legend()
    plt.show()
    return fig
