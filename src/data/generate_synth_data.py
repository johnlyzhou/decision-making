from typing import Tuple, Type

import numpy as np
from tqdm import tqdm

from src.data.agents import AgentInterface, QLearningAgent, InferenceAgent
from src.data.environments import EnvironmentInterface, DynamicForagingTask
from src.data.experiments import SynthExperiment
from src.utils import build_config, blockify, normalize_choice_block_side, truncate_blocks


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
    :param save: Whether to save results or not.
    :return: Two arrays, trial array of shape (num_trials_per_block, num_samples) and label array of shape
    (labels_dims, num_samples) containing parameter information.
    """
    if task == DynamicForagingTask:
        task_name = "DynamicForagingTask"
    else:
        raise NotImplementedError

    if agent == QLearningAgent:
        agent_name = "QLearningAgent"
        p1_bounds = lr_bounds
        p2_bounds = eps_bounds
        num_p1 = num_lrs
        num_p2 = num_eps
    elif agent == InferenceAgent:
        agent_name = "InferenceAgent"
        p1_bounds = pswitch_bounds
        p2_bounds = prew_bounds
        num_p1 = num_pswitches
        num_p2 = num_prews
    else:
        raise NotImplementedError

    p1s = np.linspace(p1_bounds[0], p1_bounds[1], num=num_p1)
    p2s = np.linspace(p2_bounds[0], p2_bounds[1], num=num_p2)
    block_choices = np.zeros((num_p1 * num_p2 * num_blocks, trial_bounds[0]))
    labels = np.zeros((num_p1 * num_p2 * num_blocks, 2))
    running_idx = 0

    for p1 in tqdm(p1s):
        for p2 in p2s:
            if agent == QLearningAgent:
                config = build_config(task_name, agent_name, num_blocks, trial_bounds, true_pr_rew,
                                      lr=float(p1), eps=float(p2))
            elif agent == InferenceAgent:
                config = build_config(task_name, agent_name, num_blocks, trial_bounds, true_pr_rew,
                                      pr_switch=float(p1), pr_rew=float(p2))
            else:
                raise NotImplementedError
            expt = SynthExperiment(config)
            expt.run()
            actions = expt.agent.action_history
            rewards = expt.environment.reward_history
            blocked_actions = blockify(expt.blocks, actions)
            blocks = expt.blocks
            normalized_actions = [normalize_choice_block_side(blocked_actions[block_idx], side=blocks[block_idx][0])
                                  for block_idx in range(len(blocks))]
            truncated_actions = truncate_blocks(normalized_actions, truncate_length=trial_bounds[0])

            for action_block in truncated_actions:
                block_choices[running_idx, :] = np.array(action_block)
                labels[running_idx, :] = np.array([p1, p2])
                running_idx += 1

    return block_choices, labels
