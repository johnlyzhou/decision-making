

def get_block_indices(blocks):
    """Return list of indices of the first and last trials of each block within all trials of the experiment."""
    indices = []
    trial_idx = 0
    for i in range(len(blocks)):
        start = trial_idx
        end = trial_idx + blocks[i][2]
        indices.append((start, end))
        trial_idx = end
    return indices


def blockify(blocks, obs):
    """Partition a list of trial observations into a list of blocks of trial observations."""
    indices = get_block_indices(blocks)
    if sum([block[2] for block in blocks]) != len(obs):
        raise ValueError("Observation length doesn't match block lengths!")
    return [obs[start:end] for start, end in indices]

def make_side_agnostic(blocks, obs):
    """Make blocks of observations side-agnostic by flipping all blocks of trials to make the """
