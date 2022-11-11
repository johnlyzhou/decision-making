import numpy as np

from src.features.fit_curves import epsilon_sigmoid
from src.visualization.plot_replications import plot_fitted_block, plot_sigmoids


class ExperimentData:
    """The purpose of this class is to keep consistent indexing across original data and any transformations,
    allowing us to sanity check and inspect the results of downstream analyses."""
    def __init__(self, expt_name, repo_dir):
        self.expt_name = expt_name
        self.repo_dir = repo_dir
        self.data_path = f"{repo_dir}/data/processed/{expt_name}"
        self.choice_blocks = None
        self.agent_labels = None
        self.parameter_labels = None
        self.sigmoid_parameters = None
        self.foraging_efficiency = None
        self.__load_data()

        # Store a feature embedding here
        self.embedding = None

    def __load_data(self):
        self.choice_blocks = np.load(f"{self.data_path}/choice_blocks.npy")
        self.agent_labels = np.load(f"{self.data_path}/agent_labels.npy")
        self.parameter_labels = np.load(f"{self.data_path}/parameter_labels.npy")
        self.sigmoid_parameters = np.load(f"{self.data_path}/sigmoid_parameters.npy")
        self.foraging_efficiency = np.load(f"{self.data_path}/foraging_efficiency.npy")

    def build_modeling_feats(self,
                             feat_path: str = None,
                             include_sigmoid: bool = True,
                             include_feff: bool = False,
                             include_block: bool = False):
        feat_list = []

        if include_sigmoid:
            feat_list.append(self.sigmoid_parameters)
        if include_feff:
            feat_list.append(self.foraging_efficiency)
        if include_block:
            feat_list.append(self.choice_blocks)

        feats = np.hstack(feat_list)

        if not feat_path:
            np.save(f"{self.data_path}/modeling_features.npy")
        else:
            np.save(feat_path, feats)

        return feats

    def visualize_block(self, idx):
        plot_fitted_block(self.choice_blocks[idx], epsilon_sigmoid, tuple(self.sigmoid_parameters[idx]))

    def visualize_sigmoids(self, idxs):
        params_list = self.parameter_labels[idxs]
        plot_sigmoids(epsilon_sigmoid, params_list)
