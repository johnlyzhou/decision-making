import numpy as np

from src.data.generate_synth_data import run_experiment_batch
from src.data.agents import QLearningAgent, InferenceAgent
from src.data.environments import DynamicForagingTask
from src.features.fit_curves import get_sigmoid_feats
from src.features.build_features import compute_foraging_efficiency
from src.features.losses import mse_loss, binary_nll

if __name__ == "__main__":
    data_dir = "/Users/johnzhou/research/decision-making/data"

    # Q Learning agent
    block_choices, labels = run_experiment_batch(DynamicForagingTask, QLearningAgent, num_blocks=100, save=True)
    np.save(f"{data_dir}/interim/ql_trials.npy", block_choices)
    print(block_choices.shape)
    E = compute_foraging_efficiency(block_choices,
                                    save_path=f"{data_dir}/processed/ql_eff.npy")
    print(E.shape)
    nll_sigmoid_feats = get_sigmoid_feats(block_choices, binary_nll, plot=False, save=True,
                                          save_path=f"{data_dir}/processed/ql_nll_sig.npy")
    print(nll_sigmoid_feats.shape)
    mse_sigmoid_feats = get_sigmoid_feats(block_choices, mse_loss, plot=False, save=True,
                                          save_path=f"{data_dir}/processed/ql_mse_sig.npy")
    print(mse_sigmoid_feats.shape)

    # Inference agent
    block_choices, labels = run_experiment_batch(DynamicForagingTask, InferenceAgent, num_blocks=100, save=True)
    np.save(f"{data_dir}/interim/inf_trials.npy", block_choices)
    print(block_choices.shape)
    E = compute_foraging_efficiency(block_choices,
                                    save_path=f"{data_dir}/processed/inf_eff.npy")
    print(E.shape)
    nll_sigmoid_feats = get_sigmoid_feats(block_choices, binary_nll, plot=False, save=True,
                                          save_path=f"{data_dir}/processed/inf_nll_sig.npy")
    print(nll_sigmoid_feats.shape)
    mse_sigmoid_feats = get_sigmoid_feats(block_choices, mse_loss, plot=False, save=True,
                                          save_path=f"{data_dir}/processed/inf_mse_sig.npy")
    print(mse_sigmoid_feats.shape)

    # Real agent
    block_choices = np.load(f"{data_dir}/interim/real_trials.npy")
    print(block_choices.shape)
    E = compute_foraging_efficiency(block_choices,
                                    save_path=f"{data_dir}/processed/real_eff.npy")
    print(E.shape)
    nll_sigmoid_feats = get_sigmoid_feats(block_choices, binary_nll, plot=False, save=True,
                                          save_path=f"{data_dir}/processed/real_nll_sig.npy")
    print(nll_sigmoid_feats.shape)
    mse_sigmoid_feats = get_sigmoid_feats(block_choices, mse_loss, plot=False, save=True,
                                          save_path=f"{data_dir}/processed/real_mse_sig.npy")
    print(mse_sigmoid_feats.shape)
