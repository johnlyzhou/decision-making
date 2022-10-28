from typing import Tuple, Type

import numpy as np
import scipy
from tqdm import tqdm

from src.data.agents import AgentInterface, QLearningAgent, InferenceAgent, SwitchingAgent
from src.data.environments import EnvironmentInterface, DynamicForagingTask
from src.data.experiments import SynthExperiment
from src.features.build_features import compute_foraging_efficiency, sigmoid_mse, sigmoid_initial_guess, sigmoid
from src.utils import build_config, blockify, normalize_choice_block_side, average_choice_blocks, truncate_blocks

INITIAL_GUESS = np.array([0.2, 5, 11])
SIGMOID_PARAM_BOUNDS = ((0, 0.5), (0, 1.4), (0, 15))


def run_experiment_batch(task: Type[EnvironmentInterface],
                         agent: Type[AgentInterface],
                         lr_bounds: Tuple[float, float] = (0.01, 1.4),
                         eps_bounds: Tuple[float, float] = (0.01, 0.5),
                         num_lrs: int = 25,
                         num_eps: int = 20,
                         pswitch_bounds: Tuple[float, float] = (0.01, 0.45),
                         prew_bounds: Tuple[float, float] = (0.55, 0.99),
                         num_pswitches: int = 25,
                         num_prews: int = 20,
                         true_pr_rew: float = 1.0,
                         trial_bounds: Tuple[int, int] = (15, 25),
                         num_blocks: int = 10):
    """
    Run a batch of experiments and return trial data and corresponding labels.
    :param task: Type of task environment.
    :param agent: Type of agent.
    :param lr_bounds: Range of possible learning rates.
    :param eps_bounds: Range of possible epsilons.
    :param num_lrs: Number of learning rate settings to run.
    :param num_eps: Number of epsilon settings to run.
    :param pswitch_bounds: Range of possible p_switches.
    :param prew_bounds: Range of possible p_rewards.
    :param num_pswitches: Number of p_switch settings to run.
    :param num_prews: Number of p_reward settings to run.
    :param true_pr_rew: Probability that the environment rewards the correct choice.
    :param trial_bounds: Range of num_trials in the block.
    :param num_blocks: Number of blocks to simulate and average across for fitting.
    :return: Two arrays, trial array of shape (num_trials_per_block, num_samples) and label array of shape
    (labels_dims, num_samples) containing parameter information.
    """
    if task == DynamicForagingTask:
        pass
    else:
        raise NotImplementedError

    if agent == QLearningAgent:
        p1_bounds = lr_bounds
        p2_bounds = eps_bounds
        num_p1 = num_lrs
        num_p2 = num_eps
    elif agent == InferenceAgent:
        p1_bounds = pswitch_bounds
        p2_bounds = prew_bounds
        num_p1 = num_pswitches
        num_p2 = num_prews
    elif agent == SwitchingAgent:
        raise NotImplementedError
    else:
        raise NotImplementedError

    p1s = np.linspace(p1_bounds[0], p1_bounds[1], num=num_p1)
    p2s = np.linspace(p2_bounds[0], p2_bounds[1], num=num_p2)
    block_feats = np.zeros((trial_bounds[0], num_p1 * num_p2 * num_blocks))
    labels = np.zeros((2, num_p1 * num_p2 * num_blocks))
    failed_fits = 0
    running_idx = 0

    for p1 in tqdm(p1s):
        for p2 in p2s:
            if agent == QLearningAgent:
                config = build_config("DynamicForagingTask", "QLearningAgent", num_blocks, trial_bounds, true_pr_rew,
                                      lr=float(p1), eps=float(p2))
            elif agent == InferenceAgent:
                config = build_config("DynamicForagingTask", "QLearningAgent", num_blocks, trial_bounds, true_pr_rew,
                                      pr_switch=float(p1), pr_rew=float(p2))
            else:
                raise NotImplementedError

            expt = SynthExperiment(config)
            expt.run()
            actions = expt.agent.action_history
            rewards = expt.environment.reward_history
            E = compute_foraging_efficiency(actions, rewards)
            blocked_actions = blockify(expt.blocks, actions)
            blocks = expt.blocks
            normalized_actions = [normalize_choice_block_side(blocked_actions[block_idx], blocks[block_idx][0])
                                  for block_idx in range(len(blocks))]
            truncated_actions = truncate_blocks(normalized_actions, truncate_length=trial_bounds[0])

            for action_block in truncated_actions:
                block_feats[:, running_idx] = np.array(action_block)
                labels[:, running_idx] = np.array([p1, p2])
                running_idx += 1

    return block_feats, labels


def qlearning_grid_sigmoid_fits(lr_bounds=(0.01, 1.4), eps_bounds=(0.01, 0.5), num_lrs=25, num_eps=20,
                                true_pr_rew=1.0, trial_bounds=(15, 25), num_blocks=1000):
    epsilons = np.linspace(eps_bounds[0], eps_bounds[1], num=num_eps)
    lrs = np.linspace(lr_bounds[0], lr_bounds[1], num=num_lrs)
    trial_feats = np.zeros((15, num_lrs * num_eps))
    sig_feats = np.zeros((4, num_lrs * num_eps))
    labels = np.zeros((2, num_lrs * num_eps))
    failed_fits = 0
    running_idx = 0
    for lr in tqdm(lrs):
        for eps in epsilons:
            config = build_config("DynamicForagingTask", "QLearningAgent", num_blocks, trial_bounds, true_pr_rew, lr=float(lr),
                                  eps=float(eps))
            expt = SynthExperiment(config)
            expt.run()
            actions = expt.agent.action_history
            rewards = expt.environment.reward_history
            E = compute_foraging_efficiency(actions, rewards)
            blocked_actions = blockify(expt.blocks, actions)
            blocks = expt.blocks
            normalized_actions = [normalize_choice_block_side(blocked_actions[block_idx], blocks[block_idx][0])
                                  for block_idx in range(len(blocks))]
            max_len = max([block[2] for block in blocks])
            averaged_blocks = average_choice_blocks(normalized_actions, max_len=max_len, mode='truncate')

            x_obs = np.float64(np.arange(len(averaged_blocks)))
            y_obs = np.float64(np.array(averaged_blocks))

            trial_feats[:, running_idx] = y_obs
            try:
                params = scipy.optimize.minimize(sigmoid_mse, sigmoid_initial_guess(y_obs), args=(x_obs, y_obs),
                                                 bounds=SIGMOID_PARAM_BOUNDS).x
                sig_feats[:, running_idx] = np.array([*params, E])
            except RuntimeError:
                failed_fits += 1
            labels[:, running_idx] = np.array([eps, lr])
            running_idx += 1
    print(failed_fits)
    np.save("/Users/johnzhou/research/decision-making/data/synth/ql_sig_trial_feats.npy", trial_feats)
    np.save("/Users/johnzhou/research/decision-making/data/synth/ql_sig_feats.npy", sig_feats)
    np.save("/Users/johnzhou/research/decision-making/data/synth/ql_sig_labels.npy", labels)


def inference_grid_sigmoid_fits(pswitch_bounds=(0.01, 0.45), prew_bounds=(0.55, 0.99), num_pswitches=25, num_prews=20,
                                true_pr_rew=1.0, trial_bounds=(15, 25), num_blocks=1000):
    pswitches = np.linspace(pswitch_bounds[0], pswitch_bounds[1], num=num_pswitches)
    prews = np.linspace(prew_bounds[0], prew_bounds[1], num=num_prews)
    sig_feats = np.zeros((4, num_pswitches * num_prews))
    labels = np.zeros((2, num_pswitches * num_prews))
    trial_feats = np.zeros((15, num_pswitches * num_prews))
    failed_fits = 0
    running_idx = 0
    for pswitch in tqdm(pswitches):
        for prew in prews:
            config = build_config("DynamicForagingTask", "InferenceAgent", num_blocks, trial_bounds, true_pr_rew,
                                  pr_rew=float(prew), pr_switch=float(pswitch))
            expt = SynthExperiment(config)
            expt.run()
            actions = expt.agent.action_history
            rewards = expt.environment.reward_history
            E = compute_foraging_efficiency(actions, rewards)
            blocked_actions = blockify(expt.blocks, actions)
            blocks = expt.blocks
            normalized_actions = [normalize_choice_block_side(blocked_actions[block_idx], blocks[block_idx][0])
                                  for block_idx in range(len(blocks))]
            max_len = max([block[2] for block in blocks])
            averaged_blocks = average_choice_blocks(normalized_actions, max_len=max_len, mode='truncate')

            x_obs = np.float64(np.arange(len(averaged_blocks)))
            y_obs = np.float64(np.array(averaged_blocks))

            trial_feats[:, running_idx] = y_obs
            try:
                params = scipy.optimize.minimize(sigmoid_mse, sigmoid_initial_guess(y_obs), args=(x_obs, y_obs),
                                                 bounds=SIGMOID_PARAM_BOUNDS).x
                sig_feats[:, running_idx] = np.array([*params, E])
                labels[:, running_idx] = np.array([prew, pswitch])
            except RuntimeError:
                failed_fits += 1
            labels[:, running_idx] = np.array([prew, pswitch])
            running_idx += 1
    print(failed_fits)

    np.save("/Users/johnzhou/research/decision-making/data/synth/inf_sig_trial_feats.npy", trial_feats)
    np.save("/Users/johnzhou/research/decision-making/data/synth/inf_sig_feats.npy", sig_feats)
    np.save("/Users/johnzhou/research/decision-making/data/synth/inf_sig_labels.npy", labels)
