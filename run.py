from src.data.generate_synth_data import run_experiment_batch
from src.data.agents import QLearningAgent, InferenceAgent
from src.data.environments import DynamicForagingTask
from src.features.fit_curves import epsilon_sigmoid, get_sigmoid_feats, binary_logistic, get_logistic_feats
from src.features.losses import mse_loss, binary_nll
from src.visualization.plot_replications import plot_fitted_block

if __name__ == "__main__":
    data_dir = "/Users/johnzhou/research/decision-making/data"
    block_choices, labels = run_experiment_batch(DynamicForagingTask, QLearningAgent, num_blocks=10, save=True)
    print(block_choices.shape)
    nll_sigmoid_feats = get_sigmoid_feats(block_choices, binary_nll, plot=False, save=True,
                                          save_path=f"{data_dir}/synth/ql_nll_sig.npy")
    print(nll_sigmoid_feats.shape)
    mse_sigmoid_feats = get_sigmoid_feats(block_choices, mse_loss, plot=False, save=True,
                                          save_path=f"{data_dir}/synth/ql_mse_sig.npy")
    print(mse_sigmoid_feats.shape)

    block_choices, labels = run_experiment_batch(DynamicForagingTask, InferenceAgent, num_blocks=10, save=True)
    print(block_choices.shape)
    nll_sigmoid_feats = get_sigmoid_feats(block_choices, binary_nll, plot=False, save=True,
                                          save_path=f"{data_dir}/synth/inf_nll_sig.npy")
    print(nll_sigmoid_feats.shape)
    mse_sigmoid_feats = get_sigmoid_feats(block_choices, mse_loss, plot=False, save=True,
                                          save_path=f"{data_dir}/synth/inf_mse_sig.npy")
    print(mse_sigmoid_feats.shape)
