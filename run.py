import os
from typing import Type, Tuple

import numpy as np
from numpy import ndarray
from tqdm import tqdm

from src.data.agents import QLearningAgent, InferenceAgent, BlockSwitchingAgent, AgentInterface
from src.data.environments import DynamicForagingTask, EnvironmentInterface
from src.data.experiments import SynthExperiment, BasicSynthExperiment
from src.features.build_features import generate_synth_features
from src.features.losses import mse_loss
from src.utils import normalize_choice_block_side, blockify, truncate_blocks, build_config


def model_training_run(expt_dir, task, p_rew, loss, num_blocks):
    ql_blocks, ql_params, ql_sigmoids, ql_feff = generate_synth_features(task,
                                                                         QLearningAgent,
                                                                         p_rew,
                                                                         loss,
                                                                         num_blocks=num_blocks)
    print(f"Model-free agent: {ql_blocks.shape[0]} samples, {ql_blocks.shape[1]} trials")
    inf_blocks, inf_params, inf_sigmoids, inf_feff = generate_synth_features(task,
                                                                             InferenceAgent,
                                                                             p_rew,
                                                                             loss,
                                                                             num_blocks=num_blocks)
    print(f"Model-based agent: {inf_blocks.shape[0]} samples, {inf_blocks.shape[1]} trials")

    choice_blocks = np.vstack((ql_blocks, inf_blocks))
    print(choice_blocks.shape)
    np.save(f"{expt_dir}/choice_blocks.npy", choice_blocks)

    agent_params = np.hstack((np.zeros(ql_blocks.shape[0]), np.ones(inf_blocks.shape[0])))
    print(agent_params.shape)
    np.save(f"{expt_dir}/agent_labels.npy", agent_params)

    param_labels = np.vstack((ql_params, inf_params))
    print(param_labels.shape)
    np.save(f"{expt_dir}/parameter_labels.npy", param_labels)

    sig_params = np.vstack((ql_sigmoids, inf_sigmoids))
    print(sig_params.shape)
    np.save(f"{expt_dir}/sigmoid_parameters.npy", sig_params)

    feff = np.hstack((ql_feff, inf_feff))
    print(feff.shape)
    np.save(f"{expt_dir}/foraging_efficiency.npy", feff)


def run_sampling_switching_experiment(transition_matrix: ndarray,
                                      lr_bounds: Tuple[float, float] = (0.01, 1.4),
                                      eps_bounds: Tuple[float, float] = (0.01, 0.5),
                                      num_lrs: int = 25,
                                      num_eps: int = 20,
                                      pswitch_bounds: Tuple[float, float] = (0.01, 0.45),
                                      prew_bounds: Tuple[float, float] = (0.55, 0.99),
                                      num_pswitches: int = 25,
                                      num_prews: int = 20,
                                      true_pr_rew: float = 1.0,
                                      trial_range: Tuple[int, int] = (15, 25),
                                      num_blocks: int = 50):
    pass


def run_updating_switching_experiment(transition_matrix: ndarray,
                                      lr_bounds: Tuple[float, float] = (0.01, 1.4),
                                      eps_bounds: Tuple[float, float] = (0.01, 0.5),
                                      num_lrs: int = 25,
                                      num_eps: int = 20,
                                      pswitch_bounds: Tuple[float, float] = (0.01, 0.45),
                                      prew_bounds: Tuple[float, float] = (0.55, 0.99),
                                      num_pswitches: int = 25,
                                      num_prews: int = 20,
                                      true_pr_rew: float = 1.0,
                                      trial_range: Tuple[int, int] = (15, 25),
                                      num_blocks: int = 50):
    """Switch between QL and inference agent with transition matrix, for each block randomly sample some parameter
    setting for that agent and run them through the block. May need to generate all the agents so we can update them
    all simultaneously."""
    lrs, eps = np.meshgrid(np.linspace(*lr_bounds, num=num_lrs), np.linspace(*eps_bounds, num=num_eps), indexing="ij")
    ql_agents = [QLearningAgent(lrs.flatten()[i], eps.flatten()[i], DynamicForagingTask) for i in
                 range(num_lrs * num_eps)]
    pswitches, prews = np.meshgrid(np.linspace(*pswitch_bounds, num=num_pswitches),
                                   np.linspace(*prew_bounds, num=num_prews))
    inf_agents = [InferenceAgent(pswitches.flatten()[i], prews.flatten()[i]) for i in range(num_pswitches * num_prews)]
    agent_list = [ql_agents, inf_agents]
    agent = BlockSwitchingAgent(transition_matrix, agent_list)

    blocks = []
    sides = ["LEFT", "RIGHT"]
    for idx in range(num_blocks):
        num_trials = np.random.randint(trial_range[0], trial_range[1])
        blocks.append((sides[idx % 2], true_pr_rew, num_trials))

    task = DynamicForagingTask(blocks)
    experiment = BasicSynthExperiment(agent, task)
    experiment.run()
    actions = experiment.agent.action_history
    blocked_actions = blockify(experiment.environment.blocks, actions)
    blocks = experiment.environment.blocks
    normalized_actions = [normalize_choice_block_side(blocked_actions[block_idx], side=blocks[block_idx][0])
                          for block_idx in range(len(blocks))]
    truncated_actions = truncate_blocks(normalized_actions, truncate_length=trial_range[0])
    return truncated_actions, agent.state_history


if __name__ == "__main__":
    data_dir = "/Users/johnzhou/research/decision-making/data"
    expt_name = "generate_ssm_histogram"
    environment = DynamicForagingTask
    p_reward = 1.0
    loss_func = mse_loss
    n_blocks = 1000

    # Create data directory
    experiment_dir = f"{data_dir}/processed/{expt_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    model_training_run(experiment_dir, environment, p_reward, loss_func, n_blocks)


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
