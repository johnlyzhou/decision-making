from matplotlib import pyplot as plt
import numpy as np
import ssm
from ssm.plots import gradient_cmap
from ssm.util import find_permutation
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

N_iters = 100
num_states = 2
obs_dim = len(obs[0])
time_bins = expt.environment.total_trials

hmm = ssm.HMM(num_states, obs_dim, observations="bernoulli")
# hmm = ssm.HMM(num_states, 1, observations="bernoulli")

hmm_lls = hmm.fit(obs, method="em", num_iters=N_iters, init_method="kmeans")

# plt.plot(hmm_lls, label="EM")
# # plt.plot([0, num_iters], true_ll * np.ones(2), ':k', label="True")
# plt.xlabel("EM Iteration")
# plt.ylabel("Log Probability")
# plt.legend(loc="lower right")
# plt.show()

true_states = np.array(expt.agent.state_history)
most_likely_states = hmm.most_likely_states(obs)
hmm.permute(find_permutation(true_states, most_likely_states))

# Plot the true and inferred discrete states
hmm_z = hmm.most_likely_states(obs)

plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.imshow(true_states[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
plt.xlim(0, time_bins)
plt.ylabel("$z_{\\mathrm{true}}$")
plt.yticks([])

plt.subplot(212)
plt.imshow(hmm_z[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1)
plt.xlim(0, time_bins)
plt.ylabel("$z_{\\mathrm{inferred}}$")
plt.yticks([])
plt.xlabel("time")

plt.tight_layout()

plt.show()

true_transition_mat = expt.transition_matrix
learned_transition_mat = hmm.transitions.transition_matrix

fig = plt.figure(figsize=(8, 4))
plt.subplot(121)
im = plt.imshow(true_transition_mat, cmap='gray')
plt.title("True Transition Matrix")

plt.subplot(122)
im = plt.imshow(learned_transition_mat, cmap='gray')
plt.title("Learned Transition Matrix")

cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()