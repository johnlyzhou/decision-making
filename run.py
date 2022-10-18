from typing import Union, Tuple

import numpy as np
import scipy
from matplotlib import pyplot as plt
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from src.data.experiments import SynthExperiment
from src.data.real_data import DynamicForagingData, normalize_block_side, pad_ragged_blocks, average_blocks
from src.features.build_features import compute_foraging_efficiency, driver_func, sigmoid
from src.utils import blockify, build_config

INITIAL_GUESS = np.array([0.1, 10, 11])
SIGMOID_PARAM_BOUNDS = ((0, 0.5), (0, float('inf')), (0, float('inf')))


def plot_block_sigmoid(eps: float = 0.1,
                       lr: float = 0.1,
                       pswitch: float = None,
                       prew: float = None,
                       trial_bounds: Tuple[int, int] = (15, 25),
                       true_pr_rew: float = 1.0,
                       num_blocks: int = 100) -> None:
    """
    Fit and plot a sigmoid curve to the average results of num_blocks trials of a Q-learning agent.
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
        config = build_config("BlockTask", "QLearningAgent", num_blocks, trial_bounds, true_pr_rew, eps=eps,
                              lr=lr)
    elif pswitch and prew:
        config = build_config("BlockTask", "InferenceAgent", num_blocks, trial_bounds, true_pr_rew, pr_switch=pswitch,
                              pr_rew=prew)
    else:
        raise ValueError("Need to specify parameters for QLearningAgent OR InferenceAgent!")

    expt = SynthExperiment(config)
    expt.run()
    actions = expt.agent.action_history
    blocked_actions = blockify(expt.blocks, actions)
    blocks = expt.blocks
    normalized_actions = [normalize_block_side(blocked_actions[block_idx], blocks[block_idx][0])
                          for block_idx in range(len(blocks))]
    max_len = max([block[2] for block in blocks])
    averaged_blocks = average_blocks(normalized_actions, max_len=max_len)

    x_obs = range(len(averaged_blocks))
    y_obs = averaged_blocks
    params = scipy.optimize.minimize(driver_func, INITIAL_GUESS, args=(x_obs, y_obs),
                                     bounds=SIGMOID_PARAM_BOUNDS).x
    plt.plot(np.linspace(0, len(averaged_blocks), num=1000),
             sigmoid(np.linspace(0, len(averaged_blocks), num=1000), *params), 'r-',
             label='fit: eps=%5.3f, k=%5.3f, x0=%5.3f' % tuple(params))
    plt.legend()
    plt.scatter(x_obs, averaged_blocks)
    plt.show()


def plot_sample_behaviors(eps: float = 0.1,
                          lr: float = 0.1,
                          pswitch: float = None,
                          prew: float = None,
                          trial_bounds: Tuple[int, int] = (15, 25),
                          true_pr_rew: float = 1.0,
                          num_blocks: int = 100) -> None:
    """
    Plot correct and incorrect choices across all blocks.
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
        config = build_config("BlockTask", "QLearningAgent", num_blocks, trial_bounds, true_pr_rew, eps=eps,
                              lr=lr)
    elif pswitch and prew:
        agent_type = "InferenceAgent"
        config = build_config("BlockTask", "InferenceAgent", num_blocks, trial_bounds, true_pr_rew, pr_switch=pswitch,
                              pr_rew=prew)
    else:
        raise ValueError("Need to specify parameters for QLearningAgent OR InferenceAgent!")

    expt = SynthExperiment(config)
    expt.run()
    actions = expt.agent.action_history
    blocked_actions = blockify(expt.blocks, actions)
    blocks = expt.blocks
    normalized_actions = [normalize_block_side(blocked_actions[block_idx], blocks[block_idx][0])
                          for block_idx in range(len(blocks))]
    max_len = max([block[2] for block in blocks])
    padded_blocks = pad_ragged_blocks(normalized_actions, max_len=max_len)

    fig, ax = plt.subplots()
    n_trials, n_blocks = padded_blocks.shape
    ys, xs = np.meshgrid(range(n_trials), range(n_blocks))
    ax.pcolormesh(ys, xs, padded_blocks.T, vmin=np.min(padded_blocks.T), vmax=np.max(padded_blocks.T),
                  cmap='coolwarm_r')
    plt.xlabel("Trials from switch")
    plt.ylabel("Block #")
    if agent_type == "QLearningAgent":
        plt.title(f"Epsilon = {eps}, Learning Rate = {lr}")
    else:
        plt.title(f"P_switch = {pswitch}, P_reward = {prew}")
    plt.show()


def q_learning_grid_run_raw(lr_bounds: Tuple[float, float] = (0.01, 1.4),
                            eps_bounds: Tuple[float, float] = (0.01, 0.5),
                            num_lrs: int = 25,
                            num_eps: int = 20,
                            true_pr_rew: float = 1.0,
                            trial_bounds: Tuple[int, int] = (15, 25),
                            num_blocks: int = 100) -> None:
    """

    :param lr_bounds: Range of possible learning rates.
    :param eps_bounds: Range of possible epsilons.
    :param num_lrs: Number of learning rate settings to run.
    :param num_eps: Number of epsilon settings to run.
    :param trial_bounds: Range of num_trials in the block.
    :param true_pr_rew: Probability that the environment rewards the correct choice.
    :param num_blocks: Number of blocks to simulate and average across for fitting.
    :return: None
    """
    epsilons = np.linspace(eps_bounds[0], eps_bounds[1], num=num_eps)
    lrs = np.linspace(lr_bounds[0], lr_bounds[1], num=num_lrs)
    feats = np.zeros((trial_bounds[0], num_lrs * num_eps * num_blocks))
    labels = np.zeros((2, num_lrs * num_eps * num_blocks))
    running_idx = 0
    for lr in tqdm(lrs):
        for eps in epsilons:
            config = build_config("BlockTask", "QLearningAgent", num_blocks, trial_bounds, true_pr_rew, lr=float(lr),
                                  eps=float(eps))
            expt = SynthExperiment(config)
            expt.run()
            actions = expt.agent.action_history
            blocked_actions = blockify(expt.blocks, actions)
            blocks = expt.blocks
            normalized_actions = [
                normalize_block_side(blocked_actions[block_idx], blocks[block_idx][0])[: trial_bounds[0]]
                for block_idx in range(len(blocks))]
            for action_block in normalized_actions:
                feats[:, running_idx] = np.array(action_block)
                labels[:, running_idx] = np.array([eps, lr])
                running_idx += 1
    np.save("/Users/johnzhou/research/decision-making/data/synth/qlearning_sim_features_raw.npy", feats)
    np.save("/Users/johnzhou/research/decision-making/data/synth/qlearning_sim_labels_raw.npy", labels)


def inference_grid_run_raw(pswitch_bounds: Tuple[float, float] = (0.01, 0.45),
                           prew_bounds: Tuple[float, float] = (0.55, 0.99),
                           num_pswitches: int = 25,
                           num_prews: int = 20,
                           true_pr_rew: float = 1.0,
                           trial_bounds: Tuple[int, int] = (15, 25),
                           num_blocks: int = 100) -> None:
    """

    :param pswitch_bounds: Range of possible p_switches.
    :param prew_bounds: Range of possible p_rewards.
    :param num_pswitches: Number of p_switch settings to run.
    :param num_prews: Number of p_reward settings to run.
    :param trial_bounds: Range of num_trials in the block.
    :param true_pr_rew: Probability that the environment rewards the correct choice.
    :param num_blocks: Number of blocks to simulate and average across for fitting.
    :return: None
    """
    pswitches = np.linspace(pswitch_bounds[0], pswitch_bounds[1], num=num_pswitches)
    prews = np.linspace(prew_bounds[0], prew_bounds[1], num=num_prews)
    feats = np.zeros((trial_bounds[0], num_pswitches * num_prews * num_blocks))
    labels = np.zeros((2, num_pswitches * num_prews * num_blocks))
    running_idx = 0
    for prew in tqdm(prews):
        for pswitch in pswitches:
            config = build_config("BlockTask", "InferenceAgent", num_blocks, trial_bounds, true_pr_rew,
                                  pr_rew=float(prew), pr_switch=float(pswitch))
            expt = SynthExperiment(config)
            expt.run()
            actions = expt.agent.action_history
            blocked_actions = blockify(expt.blocks, actions)
            blocks = expt.blocks
            normalized_actions = [
                normalize_block_side(blocked_actions[block_idx], blocks[block_idx][0])[:trial_bounds[0]]
                for block_idx in range(len(blocks))]
            for action_block in normalized_actions:
                feats[:, running_idx] = np.array(action_block)
                labels[:, running_idx] = np.array([pswitch, prew])
                running_idx += 1

    np.save("/Users/johnzhou/research/decision-making/data/synth/inference_sim_features_raw.npy", feats)
    np.save("/Users/johnzhou/research/decision-making/data/synth/inference_sim_labels_raw.npy", labels)


def qlearning_grid_sigmoid_fits(lr_bounds=(0.01, 1.4), eps_bounds=(0.01, 0.5), num_lrs=25, num_eps=20,
                                true_pr_rew=1.0, trial_bounds=(15, 25), num_blocks=1000):
    epsilons = np.linspace(eps_bounds[0], eps_bounds[1], num=num_eps)
    lrs = np.linspace(lr_bounds[0], lr_bounds[1], num=num_lrs)
    feats = np.zeros((4, num_lrs * num_eps))
    labels = np.zeros((2, num_lrs * num_eps))
    failed_fits = 0
    running_idx = 0
    for lr in tqdm(lrs):
        for eps in epsilons:
            config = build_config("BlockTask", "QLearningAgent", num_blocks, trial_bounds, true_pr_rew, lr=float(lr),
                                  eps=float(eps))
            expt = SynthExperiment(config)
            expt.run()
            actions = expt.agent.action_history
            rewards = expt.environment.__reward_history
            E = compute_foraging_efficiency(actions, rewards)
            blocked_actions = blockify(expt.blocks, actions)
            blocks = expt.blocks
            normalized_actions = [normalize_block_side(blocked_actions[block_idx], blocks[block_idx][0])
                                  for block_idx in range(len(blocks))]
            max_len = max([block[2] for block in blocks])
            averaged_blocks = average_blocks(normalized_actions, max_len=max_len, mode='truncate')

            x_obs = range(len(averaged_blocks))
            y_obs = averaged_blocks
            try:
                params = scipy.optimize.minimize(driver_func, INITIAL_GUESS, args=(x_obs, y_obs),
                                                 bounds=SIGMOID_PARAM_BOUNDS).x
                feats[:, running_idx] = np.array([*params, E])
                labels[:, running_idx] = np.array([eps, lr])
            except RuntimeError:
                failed_fits += 1
            labels[:, running_idx] = np.array([eps, lr])
            running_idx += 1
    print(failed_fits)
    np.save("/Users/johnzhou/research/decision-making/data/synth/qlearning_sim_features.npy", feats)
    np.save("/Users/johnzhou/research/decision-making/data/synth/qlearning_sim_labels.npy", labels)


def inference_grid_sigmoid_fits(pswitch_bounds=(0.01, 0.45), prew_bounds=(0.55, 0.99), num_pswitches=25, num_prews=20,
                                true_pr_rew=1.0, trial_bounds=(15, 25), num_blocks=1000):
    pswitches = np.linspace(pswitch_bounds[0], pswitch_bounds[1], num=num_pswitches)
    prews = np.linspace(prew_bounds[0], prew_bounds[1], num=num_prews)
    feats = np.zeros((4, num_pswitches * num_prews))
    labels = np.zeros((2, num_pswitches * num_prews))
    running_idx = 0
    for prew in tqdm(prews):
        for pswitch in pswitches:
            config = build_config("BlockTask", "InferenceAgent", num_blocks, trial_bounds, true_pr_rew,
                                  pr_rew=float(prew), pr_switch=float(pswitch))
            expt = SynthExperiment(config)
            expt.run()
            actions = expt.agent.action_history
            rewards = expt.environment.__reward_history
            E = compute_foraging_efficiency(actions, rewards)
            blocked_actions = blockify(expt.blocks, actions)
            blocks = expt.blocks
            normalized_actions = [normalize_block_side(blocked_actions[block_idx], blocks[block_idx][0])
                                  for block_idx in range(len(blocks))]
            max_len = max([block[2] for block in blocks])
            averaged_blocks = average_blocks(normalized_actions, max_len=max_len, mode='truncate')

            x_obs = range(len(averaged_blocks))
            y_obs = averaged_blocks
            params = scipy.optimize.minimize(driver_func, INITIAL_GUESS, args=(x_obs, y_obs),
                                             bounds=SIGMOID_PARAM_BOUNDS).x
            feats[:, running_idx] = np.array([*params, E])
            labels[:, running_idx] = np.array([pswitch, prew])
            running_idx += 1

    np.save("/Users/johnzhou/research/decision-making/data/synth/inference_sim_features.npy", feats)
    np.save("/Users/johnzhou/research/decision-making/data/synth/inference_sim_labels.npy", labels)


if __name__ == "__main__":
    q_learning_grid_run_raw()
    inference_grid_run_raw()
    # parser = argparse.ArgumentParser(description='Run a simulated 2AFC task and fit a state space model.')
    # parser.add_argument('config', help='A required path to experiment configuration file.')
    # args = parser.parse_args()
    # test_run(args.config)
