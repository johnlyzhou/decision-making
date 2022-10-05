from matplotlib import pyplot as plt
from ssm.plots import gradient_cmap
import seaborn as sns

sns.set_style("white")
sns.set_context("talk")

color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange"
    ]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)


def plot_em_fitting(hmm_lls):
    plt.plot(hmm_lls, label="EM")
    plt.xlabel("EM Iteration")
    plt.ylabel("Log Probability")
    plt.legend(loc="lower right")
    plt.show()


def plot_true_vs_inferred_states(true_states, inferred_states):
    time_bins = len(true_states)
    fig = plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.imshow(true_states[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    plt.xlim(0, time_bins)
    plt.ylabel("$z_{\\mathrm{true}}$")
    plt.yticks([])

    plt.subplot(212)
    plt.imshow(inferred_states[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
    plt.xlim(0, time_bins)
    plt.ylabel("$z_{\\mathrm{inferred}}$")
    plt.yticks([])
    plt.xlabel("time")

    plt.tight_layout()
    plt.show()

    return fig


def plot_true_vs_inferred_transition_matrix(true_transition_matrix, inferred_transition_matrix):
    fig = plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.imshow(true_transition_matrix, cmap='gray')
    plt.title("True Transition Matrix")

    plt.subplot(122)
    im = plt.imshow(inferred_transition_matrix, cmap='gray')
    plt.title("Learned Transition Matrix")

    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    return fig