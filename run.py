from src.data.generate_synth_data import qlearning_grid_solo_sigmoid_fits, qlearning_grid_sigmoid_fits, \
    inference_grid_sigmoid_fits
from src.visualization.visualize_sims import plot_behavior_simulation

if __name__ == "__main__":
    plot_behavior_simulation(eps=None,
                             lr=None,
                             pswitch=0.45,
                             prew=0.99)
