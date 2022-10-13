from typing import Union

import numpy as np
import scipy
from matplotlib import pyplot as plt
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from src.data.experiments import Experiment
from src.data.real_datasets import DynamicForagingDataset
from src.features.build_features import compute_foraging_efficiency
from src.models.fit_curves import normalize_block_ala_le2022, average_blocks, sigmoid, driver_func, \
    pad_ragged_blocks
from src.utils import blockify, build_config

COLORMAP = ['blue', 'red', 'white']


def real_sigmoid_fits(filename: str):
    real_expt = DynamicForagingDataset(filename)


def qlearning_sigmoid_fits(eps=0.1, lr=0.1, trial_bounds=(15, 25), true_pr_rew=1.0, num_blocks=100):
    config = build_config("BlockTask", "InferenceAgent", num_blocks, trial_bounds, true_pr_rew, eps=eps,
                          lr=lr)
    expt = Experiment(config)
    expt.run()
    actions = expt.agent.action_history
    rewards = expt.environment.reward_history
    E = compute_foraging_efficiency(actions, rewards)
    blocked_actions = blockify(expt.blocks, actions)
    blocks = expt.blocks
    normalized_actions = [normalize_block_ala_le2022(blocked_actions[block_idx], blocks[block_idx][0])
                          for block_idx in range(len(blocks))]
    max_len = max([block[2] for block in blocks])
    averaged_blocks = average_blocks(normalized_actions, max_len=max_len)

    x_obs = range(len(averaged_blocks))
    y_obs = averaged_blocks
    # noinspection PyTupleAssignmentBalance
    x0 = np.array([0.1, 1, 11])
    params = scipy.optimize.minimize(driver_func, x0, args=(x_obs, y_obs),
                                     bounds=((0, 0.5), (0, float('inf')), (0, float('inf')))).x
    plt.plot(np.linspace(0, len(averaged_blocks), num=1000), sigmoid(np.linspace(0, len(averaged_blocks), num=1000), *params), 'r-',
             label='fit: eps=%5.3f, k=%5.3f, x0=%5.3f' % tuple(params))
    plt.legend()
    plt.scatter(x_obs, averaged_blocks)
    plt.show()


def inference_sigmoid_fits(pswitch=0.15666666666666668, prew=0.55, trial_bounds=(15, 25), true_pr_rew=1.0, num_blocks=100):
    config = build_config("BlockTask", "InferenceAgent", num_blocks, trial_bounds, true_pr_rew, pr_switch=pswitch,
                          pr_rew=prew)
    expt = Experiment(config)
    expt.run()
    actions = expt.agent.action_history
    rewards = expt.environment.reward_history
    E = compute_foraging_efficiency(actions, rewards)
    blocked_actions = blockify(expt.blocks, actions)
    blocks = expt.blocks
    normalized_actions = [normalize_block_ala_le2022(blocked_actions[block_idx], blocks[block_idx][0])
                          for block_idx in range(len(blocks))]
    max_len = max([block[2] for block in blocks])
    averaged_blocks = average_blocks(normalized_actions, max_len=max_len, mode='truncate')

    x_obs = range(len(averaged_blocks))
    y_obs = averaged_blocks
    # noinspection PyTupleAssignmentBalance
    x0 = np.array([0.1, 1, 11])
    params = scipy.optimize.minimize(driver_func, x0, args=(x_obs, y_obs),
                                     bounds=((0, 0.5), (0, float('inf')), (0, float('inf')))).x
    plt.plot(np.linspace(0, len(averaged_blocks), num=1000), sigmoid(np.linspace(0, len(averaged_blocks), num=1000), *params), 'r-',
             label='fit: eps=%5.3f, k=%5.3f, x0=%5.3f' % tuple(params))
    plt.legend()
    plt.scatter(range(len(averaged_blocks)), averaged_blocks)
    plt.show()


def qlearning_bluered_run(lr=1.2, eps=0.2, trial_bounds=(15, 25), true_pr_rew=1.0, num_blocks=100):
    config = build_config("BlockTask", "QLearningAgent", num_blocks, trial_bounds, true_pr_rew, lr=lr,
                          eps=eps)
    expt = Experiment(config)
    expt.run()
    actions = expt.agent.action_history
    rewards = expt.environment.reward_history
    E = compute_foraging_efficiency(actions, rewards)
    blocked_actions = blockify(expt.blocks, actions)
    blocks = expt.blocks
    normalized_actions = [normalize_block_ala_le2022(blocked_actions[block_idx], blocks[block_idx][0])
                          for block_idx in range(len(blocks))]
    max_len = max([block[2] for block in blocks])
    padded_blocks = pad_ragged_blocks(normalized_actions, max_len=max_len)

    fig, ax = plt.subplots()
    n_trials, n_blocks = padded_blocks.shape
    ys, xs = np.meshgrid(range(n_trials), range(n_blocks))
    c = ax.pcolormesh(ys, xs, padded_blocks.T, vmin=np.min(padded_blocks.T), vmax=np.max(padded_blocks.T), cmap='coolwarm_r')
    # colors = ["white", "blue", "red"]
    # for i in range(xs.size):
    #     # print(padded_blocks.shape)
    #     print(xs.shape)
    #     # print(padded_blocks[ys[i], xs[i]])
    #     plt.scatter(xs[i], ys[i], color=colors[1])
    plt.show()


def inference_bluered_run(pswitch=0.45, prew=0.99, trial_bounds=(15, 25), true_pr_rew=1.0, num_blocks=100):
    config = build_config("BlockTask", "InferenceAgent", num_blocks, trial_bounds, true_pr_rew, pr_switch=pswitch,
                          pr_rew=prew)
    expt = Experiment(config)
    expt.run()
    actions = expt.agent.action_history
    rewards = expt.environment.reward_history
    E = compute_foraging_efficiency(actions, rewards)
    blocked_actions = blockify(expt.blocks, actions)
    blocks = expt.blocks
    normalized_actions = [normalize_block_ala_le2022(blocked_actions[block_idx], blocks[block_idx][0])
                          for block_idx in range(len(blocks))]
    max_len = max([block[2] for block in blocks])
    padded_blocks = pad_ragged_blocks(normalized_actions, max_len=max_len)

    fig, ax = plt.subplots()
    n_trials, n_blocks = padded_blocks.shape
    xs, ys = np.meshgrid(range(n_blocks), range(n_trials))
    c = ax.pcolormesh(xs, ys, padded_blocks, vmin=np.min(padded_blocks), vmax=np.max(padded_blocks))
    # colors = ["white", "blue", "red"]
    # for i in range(xs.size):
    #     # print(padded_blocks.shape)
    #     print(xs.shape)
    #     # print(padded_blocks[ys[i], xs[i]])
    #     plt.scatter(xs[i], ys[i], color=colors[1])
    plt.show()


def q_learning_grid_run(lr_bounds=(0.01, 1.4), eps_bounds=(0.01, 0.5), num_lrs=25, num_eps=20,
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
            expt = Experiment(config)
            expt.run()
            actions = expt.agent.action_history
            rewards = expt.environment.reward_history
            E = compute_foraging_efficiency(actions, rewards)
            blocked_actions = blockify(expt.blocks, actions)
            blocks = expt.blocks
            normalized_actions = [normalize_block_ala_le2022(blocked_actions[block_idx], blocks[block_idx][0])
                                  for block_idx in range(len(blocks))]
            max_len = max([block[2] for block in blocks])
            averaged_blocks = average_blocks(normalized_actions, max_len=max_len, mode='truncate')

            x_obs = range(len(averaged_blocks))
            y_obs = averaged_blocks
            try:
                # noinspection PyTupleAssignmentBalance
                popt, pcov = scipy.optimize.curve_fit(sigmoid, x_obs, y_obs, (0.1, 1, 11),
                                                      bounds=(0, [0.5, float('inf'), float('inf')]))
                feats[:, running_idx] = np.array([*popt, E])
                labels[:, running_idx] = np.array([eps, lr])
            except RuntimeError:
                failed_fits += 1
            labels[:, running_idx] = np.array([eps, lr])
            running_idx += 1
    print(failed_fits)
    np.save("/Users/johnzhou/research/decision-making/data/qlearning_sim_features.npy", feats)
    np.save("/Users/johnzhou/research/decision-making/data/qlearning_sim_labels.npy", labels)


def inference_grid_run(pswitch_bounds=(0.01, 0.45), prew_bounds=(0.55, 0.99), num_pswitches=25, num_prews=20,
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
            expt = Experiment(config)
            expt.run()
            actions = expt.agent.action_history
            rewards = expt.environment.reward_history
            E = compute_foraging_efficiency(actions, rewards)
            blocked_actions = blockify(expt.blocks, actions)
            blocks = expt.blocks
            normalized_actions = [normalize_block_ala_le2022(blocked_actions[block_idx], blocks[block_idx][0])
                                  for block_idx in range(len(blocks))]
            max_len = max([block[2] for block in blocks])
            averaged_blocks = average_blocks(normalized_actions, max_len=max_len, mode='truncate')

            x_obs = range(len(averaged_blocks))
            y_obs = averaged_blocks
            # noinspection PyTupleAssignmentBalance
            x0 = np.array([0.1, 1, 11])
            params = scipy.optimize.minimize(driver_func, x0, args=(x_obs, y_obs),
                                             bounds=((0, 0.5), (0, float('inf')), (0, float('inf')))).x
            feats[:, running_idx] = np.array([*params, E])
            labels[:, running_idx] = np.array([pswitch, prew])
            running_idx += 1

    np.save("/Users/johnzhou/research/decision-making/data/inference_sim_features.npy", feats)
    np.save("/Users/johnzhou/research/decision-making/data/inference_sim_labels.npy", labels)


def test_run(config: Union[str, DictConfig]) -> None:
    config = build_config("BlockTask", "QLearningAgent", 100, (15, 25), 1.0,
                          lr=float(0.10), eps=float(0.10))
    # Run a simulated experiment according to the config file settings
    expt = Experiment(config)
    expt.run()

    actions = expt.agent.action_history
    rewards = expt.environment.reward_history
    # beliefs = expt.agent.side_belief_history
    # plt.figure()
    # plt.plot(range(len(beliefs)), [belief[0] for belief in beliefs])
    # plt.plot(range(len(beliefs)), [belief[1] for belief in beliefs])
    # plt.show()
    E = compute_foraging_efficiency(actions, rewards)
    blocked_actions = blockify(expt.blocks, actions)
    blocks = expt.blocks
    normalized_actions = [normalize_block_ala_le2022(blocked_actions[block_idx], blocks[block_idx][0])
                          for block_idx in range(len(blocks))]
    max_len = max([block[2] for block in blocks])
    averaged_blocks = average_blocks(normalized_actions, max_len=max_len)

    x_obs = range(len(averaged_blocks))
    y_obs = averaged_blocks

    x0 = np.array([0.1, 1, 11])
    res = scipy.optimize.minimize(driver_func, x0, args=(x_obs, y_obs), bounds=((0, 0.5), (0, float('inf')), (0, float('inf'))))
    print(res.x)

    # noinspection PyTupleAssignmentBalance
    # popt, pcov = scipy.optimize.curve_fit(sigmoid, x_obs, y_obs, (0.1, 1, 11),
    #                                       bounds=(0, [0.5, float('inf'), float('inf')]))
    plt.plot(x_obs, sigmoid(x_obs, *res.x), 'r-',
             label='fit: eps=%5.3f, k=%5.3f, x0=%5.3f' % tuple(res.x))
    plt.legend()
    plt.scatter(range(len(averaged_blocks)), averaged_blocks)
    plt.show()

    # Verify correctness of normalized actions
    # bounds = []
    # plt.figure(figsize=(10, 1))
    # for block in blocked_actions:
    #     if len(bounds) == 0:
    #         bounds.append(len(block))
    #     else:
    #         bounds.append(bounds[-1] + len(block))
    #     plt.axvline(x=bounds[-1], color="black")
    # normalized_actions = list(itertools.chain(*normalized_actions))
    # plt.scatter(range(len(normalized_actions)), normalized_actions, s=15)
    # plt.xlim([0, len(normalized_actions)])
    # plt.show()


if __name__ == "__main__":
    real_sigmoid_fits("/Users/johnzhou/research/decision-making/data/DynamicForaging/MR15_DynamicForaging_20221006_125935.mat")
    # parser = argparse.ArgumentParser(description='Run a simulated 2AFC task and fit a state space model.')
    # parser.add_argument('config', help='A required path to experiment configuration file.')
    # args = parser.parse_args()
    # test_run(args.config)
