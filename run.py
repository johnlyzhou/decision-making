import os

import numpy as np

from src.data.agents import QLearningAgent, InferenceAgent
from src.data.environments import DynamicForagingTask
from src.features.build_features import generate_synth_features
from src.features.losses import mse_loss

if __name__ == "__main__":
    data_dir = "/Users/johnzhou/research/decision-making/data"
    expt_name = "new_run"
    task = DynamicForagingTask
    p_rew = 1.0
    loss = mse_loss
    num_blocks = 100

    # Create data directory
    expt_dir = f"{data_dir}/processed/{expt_name}"
    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)

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
