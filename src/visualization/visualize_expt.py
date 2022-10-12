import itertools

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from src.data.environments import STIMULI_FREQS, STIMULI_IDXS
from src.data.experiments import Experiment
from src.utils import get_block_indices


def get_percent_left(actions: list[int], rewards: list[int], stimuli: list[int], num_stimuli: int,
                     action_type: int = None, reward_type: int = None, stimulus_type: int = None) -> ndarray:
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


def plot_psychometric_curve(blocks: list[tuple], rewards: list[int], stimuli: list[int], actions: list[int]) -> None:
    """Plot a psychometric scatter plot."""
    block_percents = []
    fig = plt.figure()
    for block_idx in range(len(blocks)):
        block_start, block_end = get_block_indices(blocks)[block_idx]
        psycho_rewards = rewards[block_start:block_end]
        psycho_stimuli = stimuli[block_start:block_end]
        psycho_actions = actions[block_start:block_end]
        percents = get_percent_left(psycho_actions, psycho_rewards, psycho_stimuli, len(STIMULI_FREQS))
        block_percents.append(percents)
        plt.scatter(STIMULI_FREQS, percents, alpha=0.5, label=f'{blocks[block_idx][0]}')
        fig.legend()
    plt.show()


def get_switching_stimuli_outcomes(expt: Experiment, match: str = "rule") -> list[list[bool]]:
    """
    Compute choice correctness for each trial with a presentation of a switching stimulus in a blockwise fashion.
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
    trial_idx = 0
    for block in blocks:
        block_switching_trials = []
        num_trials = block[2]
        left_stimuli_idxs = STIMULI_IDXS['LEFT'].copy()
        right_stimuli_idxs = STIMULI_IDXS['RIGHT'].copy()
        boundary = block[0]
        if boundary == 'HIGH':
            left_stimuli_idxs += STIMULI_IDXS['SWITCH']
        elif boundary == 'LOW':
            right_stimuli_idxs = STIMULI_IDXS['SWITCH'] + right_stimuli_idxs

        for _ in range(num_trials):
            if stimuli[trial_idx] in STIMULI_IDXS["SWITCH"]:
                # Label an action as correct if it matches the rule based on the current boundary.
                if match == "rule":
                    block_switching_trials.append(actions[trial_idx] == (stimuli[trial_idx] in right_stimuli_idxs))
                # Label an action as correct if it matches the actual rewarded action in the environment."
                elif match == "actual":
                    block_switching_trials.append(actions[trial_idx] == rewards[trial_idx])
            trial_idx += 1
        switching_trials.append(block_switching_trials)

    return switching_trials


def plot_switching_stimuli_outcomes(switching_trials: list[list[int]]) -> None:
    bounds = []
    plt.figure(figsize=(10, 1))
    for block in switching_trials:
        if len(bounds) == 0:
            bounds.append(len(block))
        else:
            bounds.append(bounds[-1] + len(block))
        plt.axvline(x=bounds[-1], color="black")
    switching_trials = list(itertools.chain(*switching_trials))
    plt.scatter(range(len(switching_trials)), switching_trials, s=15)
    plt.xlim([0, len(switching_trials)])
    plt.show()


def plot_switching_logistic_regression(switching_trials: list[list[int]]) -> None:
    for block in switching_trials:
        X = np.asarray(range(len(block)))
        X = X[:, np.newaxis]
        y = np.asarray(block)
        clf = LogisticRegression(random_state=0).fit(X, y)
        plt.figure(1, figsize=(5, 3))
        plt.clf()
        plt.scatter(X.ravel(), y, color="black", zorder=20)
        X_test = np.linspace(0, y.size, 300)

        loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
        plt.plot(X_test, loss, color="red", linewidth=3)

        plt.ylabel("y")
        plt.xlabel("X")
        plt.ylim(0, 1)
        plt.xlim(0, y.size)
        plt.tight_layout()
        plt.show()


def plot_psychometric_scatter(expt: Experiment) -> None:
    """Plot psychometric scatter for experiment."""

    if not expt.environment.done:
        raise Exception("Run experiment before plotting results!")

    blocks = expt.blocks
    actions = expt.agent.action_history
    stimuli = expt.environment.stimulus_idx_history
    rewards = expt.environment.reward_history

    plot_psychometric_curve(blocks, rewards, stimuli, actions)
