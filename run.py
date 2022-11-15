import os
from typing import Type, Tuple

import numpy as np
from numpy import ndarray

from src.data.agents import QLearningAgent, InferenceAgent, BlockSwitchingAgent
from src.data.environments import DynamicForagingTask
from src.data.experiments import SynthExperiment, BasicSynthExperiment
from src.features.build_features import generate_synth_features
from src.features.losses import mse_loss
from src.utils import normalize_choice_block_side, blockify, truncate_blocks


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


def run_switching_experiment(transition_matrix: ndarray,
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
    ql_agents = [QLearningAgent(lrs.flatten()[i], eps.flatten()[i], DynamicForagingTask) for i in range(num_lrs * num_eps)]
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
    expt_name = "new_run"
    environment = DynamicForagingTask
    p_reward = 1.0
    loss_func = mse_loss
    n_blocks = 100

    # Create data directory
    experiment_dir = f"{data_dir}/processed/{expt_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    model_training_run(experiment_dir, environment, p_reward, loss_func, n_blocks)
