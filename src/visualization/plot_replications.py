from typing import Tuple, Callable, Union

from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray

from src.data.experiments import SynthExperiment
from src.utils import build_config, blockify, normalize_choice_block_side, pad_ragged_blocks

TITLES = ["Lapse", "Slope", "Offset", "Efficiency"]


def plot_fitted_block(choice_block: ndarray,
                      curve_func: Callable,
                      params: Tuple) -> None:
    """
    Plot a block of choices (can be either individual or averaged), and a fitted curve on top.
    :param choice_block: Single block of binary trial choices.
    :param curve_func: Curve function, e.g. sigmoid with epsilon parameter or logistic.
    :param params: Parameters for curve function.
    :return: None
    """
    x_bounds = (0, choice_block.shape[1])

    plt.plot(np.linspace(*x_bounds, num=1000),
             curve_func(np.linspace(*x_bounds, num=1000), *params),
             label=f"Parameters: {params}")
    plt.scatter(range(choice_block.size), list(choice_block))
    plt.xlim(x_bounds)
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


def plot_sigmoids(curve_func: Callable,
                  params_list: Union[ndarray, list],
                  xlim: Tuple[int, int] = (0, 14),
                  ylim: Tuple[int, int] = (0, 1),
                  num_samples: int = 1000) -> None:
    if type(params_list) == ndarray:
        if params_list.ndim() == 1:
            params_list = [params_list]
        if params_list.ndim() > 2:
            raise ValueError("Params should be of shape (num_samples, block_length).")
    else:
        if type(params_list[0]) != list or type(params_list[0]) != ndarray:
            params_list = [params_list]

    for params in params_list:
        plt.plot(np.linspace(*xlim, num=num_samples),
                 curve_func(np.linspace(*xlim, num=num_samples), *params),
                 label=f"Parameters: {params}")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.show()


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
    :param feats: Parameters from fitting sigmoid curves to averaged blocks of trials.
    :param feat_idx: Index of feature to be heatmapped (one of TITLES).
    :param xrange: Range of values for x, a parameter of the agent (typically epsilon or Prew).
    :param yrange: Range of values for y, a parameter of the agent (typically learning rate or Pswitch).
    :param xnum: Number of evenly spaced x values within xrange.
    :param ynum: Number of evenly spaced y values within yrange.
    :param xlabel: Label of x parameter for plotting.
    :param ylabel: Label of y parameter for plotting.
    :param transpose: Whether to transpose feature matrix after reshaping (useful when you've switched epsilon and
    learning rate or Prew and Pswitch in the order of inner and outer loops when generating data.
    :return: None
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
