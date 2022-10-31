import matplotlib.pyplot as plt

from src.data.environments import STIMULI_FREQS
from src.utils import get_block_indices


def plot_action_values(stimulus_action_value_history, blocks):
    """
    Plot left action-values of Q-learning agent over time, normalized to [0, 1] (so right action-value is 1 - left
    action-value).
    :param blocks: block parameters for experiment.
    :param stimulus_action_value_history: a list of lists of numpy arrays, where each numpy array is the action-values
    for a particular stimulus and each inner list is the action-values for all stimuli at that timestep.
    :return:
    """
    block_idxs = get_block_indices(blocks)
    fig = plt.figure()
    for i in block_idxs:
        plt.axvline(x=i[1], color="black")
    for stimulus_idx in range(len(STIMULI_FREQS)):
        left_action_values = [trial[stimulus_idx][0] for trial in stimulus_action_value_history]
        plt.plot(left_action_values, label=f'Stimulus: {stimulus_idx}')
        fig.legend()
    plt.xlim([0, len(stimulus_action_value_history)])
    plt.axhline(y=0.5, color="black", linestyle='--')
    plt.show()
