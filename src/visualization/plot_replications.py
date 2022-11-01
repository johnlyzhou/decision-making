from typing import Tuple

import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy import ndarray

from src.data.experiments import SynthExperiment
from src.data.generate_synth_data import INITIAL_GUESS, SIGMOID_PARAM_BOUNDS
from src.features.fit_curves import epsilon_sigmoid
from src.features.losses import mse_loss
from src.utils import build_config, blockify, normalize_choice_block_side, average_choice_blocks, pad_ragged_blocks

TITLES = ["Lapse", "Slope", "Offset", "Efficiency"]


def plot_heat_map(feats: ndarray,
                  feat_idx: int,
                  xrange: Tuple[float, float],
                  yrange: Tuple[float, float],
                  xnum: int,
                  ynum: int,
                  xlabel: str,
                  ylabel: str,
                  transpose: bool = False):
    """
    Plots heatmaps from Le et al. 2022, where the axes are parameter settings of the agents and the heat represents the
    magnitude of each fitted feature of the sigmoid curve + foraging efficiency from an average over multiple blocks
    of simualted trials.
    :param feats:
    :param feat_idx:
    :param xrange:
    :param yrange:
    :param xnum:
    :param ynum:
    :param xlabel:
    :param ylabel:
    :param transpose:
    :return:
    """
    fig, ax = plt.subplots()
    feat = feats[feat_idx, :]

    xs = np.linspace(*xrange, num=xnum)
    ys = np.linspace(*yrange, num=ynum)
    xs, ys = np.meshgrid(xs, ys)
    if transpose:
        zs = np.reshape(feat, (xnum, ynum))
        zs = zs.T
    else:
        zs = np.reshape(feat, (ynum, xnum))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    c = ax.pcolormesh(xs, ys, zs, cmap='Reds', vmin=np.min(feat), vmax=np.max(feat))
    ax.set_title(TITLES[feat_idx])
    ax.axis([*xrange, *yrange])
    fig.colorbar(c, ax=ax)


def plot_block_sigmoid(eps: float = 0.1,
                       lr: float = 0.1,
                       pswitch: float = None,
                       prew: float = None,
                       trial_bounds: Tuple[int, int] = (15, 25),
                       true_pr_rew: float = 1.0,
                       num_blocks: int = 100) -> None:
    """
    Fit and plot a sigmoid curve to the average choices of num_blocks trials of an agent.
    :param eps: Epsilon parameter for Q-learning agent.
    :param lr: Learning rate parameter for Q-learning agent.
    :param pswitch: P_switch parameter for inference agent.
    :param prew: P_reward parameter for inference agent.
    :param trial_bounds: Range of num_trials in the block.
    :param true_pr_rew: Probability that the environment rewards the correct choice.
    :param num_blocks: Number of blocks to simulate and average across for fitting.
    :return: None
    """
    if eps and lr:
        config = build_config("DynamicForagingTask", "QLearningAgent", num_blocks, trial_bounds, true_pr_rew, eps=eps,
                              lr=lr)
    elif pswitch and prew:
        config = build_config("DynamicForagingTask", "InferenceAgent", num_blocks, trial_bounds, true_pr_rew, pr_switch=pswitch,
                              pr_rew=prew)
    else:
        raise ValueError("Need to specify parameters for QLearningAgent OR InferenceAgent!")

    expt = SynthExperiment(config)
    expt.run()
    actions = expt.agent.action_history
    blocked_actions = blockify(expt.blocks, actions)
    blocks = expt.blocks
    normalized_actions = [normalize_choice_block_side(blocked_actions[block_idx], blocks[block_idx][0])
                          for block_idx in range(len(blocks))]
    max_len = max([block[2] for block in blocks])
    averaged_blocks = average_choice_blocks(normalized_actions, max_len=max_len, mode="truncate")

    x_obs = range(len(averaged_blocks))
    y_obs = averaged_blocks
    params = scipy.optimize.minimize(mse_loss, INITIAL_GUESS, method="Nelder-Mead", args=(x_obs, y_obs),
                                     bounds=SIGMOID_PARAM_BOUNDS).x
    plt.plot(np.linspace(0, len(averaged_blocks), num=1000),
             epsilon_sigmoid(np.linspace(0, len(averaged_blocks), num=1000), *params), 'r-',
             label='fit: eps=%5.3f, k=%5.3f, x0=%5.3f' % tuple(params))
    plt.legend()
    plt.scatter(x_obs, averaged_blocks)
    plt.show()


def plot_behavior_simulation(eps: float = 0.1,
                             lr: float = 0.1,
                             pswitch: float = None,
                             prew: float = None,
                             trial_bounds: Tuple[int, int] = (15, 25),
                             true_pr_rew: float = 1.0,
                             num_blocks: int = 100) -> None:
    """
    Plot correct and incorrect choices across all blocks, corresponds to Figure 3c,d in Le et al. 2022.
    :param eps: epsilon parameter for Q-learning agent.
    :param lr: learning rate parameter for Q-learning agent.
    :param pswitch: P_switch parameter for inference agent.
    :param prew: P_reward parameter for inference agent.
    :param trial_bounds: Range of num_trials in the block.
    :param true_pr_rew: Probability that the environment rewards the correct choice.
    :param num_blocks: Number of blocks to simulate and average across for fitting.
    :return: None
    """
    if eps and lr:
        agent_type = "QLearningAgent"
        config = build_config("DynamicForagingTask", "QLearningAgent", num_blocks, trial_bounds, true_pr_rew, eps=eps,
                              lr=lr)
    elif pswitch and prew:
        agent_type = "InferenceAgent"
        config = build_config("DynamicForagingTask", "InferenceAgent", num_blocks, trial_bounds, true_pr_rew, pr_switch=pswitch,
                              pr_rew=prew)
    else:
        raise ValueError("Need to specify parameters for QLearningAgent OR InferenceAgent!")

    expt = SynthExperiment(config)
    expt.run()
    actions = expt.agent.action_history
    blocked_actions = blockify(expt.blocks, actions)
    blocks = expt.blocks
    normalized_actions = [normalize_choice_block_side(blocked_actions[block_idx], blocks[block_idx][0], wrong_val=-1)
                          for block_idx in range(len(blocks))]
    max_len = max([block[2] for block in blocks])
    padded_blocks = pad_ragged_blocks(normalized_actions, max_len=max_len)

    fig, ax = plt.subplots()
    n_trials, n_blocks = padded_blocks.shape
    ys, xs = np.meshgrid(range(n_trials), range(n_blocks))

    ax.pcolormesh(ys, xs, padded_blocks.T, vmin=np.min(padded_blocks.T), vmax=np.max(padded_blocks.T),
                  cmap='bwr_r')
    plt.xlabel("Trials from switch")
    plt.ylabel("Block #")
    if agent_type == "QLearningAgent":
        plt.title(f"Epsilon = {eps}, Learning Rate = {lr}")
    else:
        plt.title(f"P_switch = {pswitch}, P_reward = {prew}")
    plt.show()
