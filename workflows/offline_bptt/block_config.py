n_timestep_frequency = 4  # number of true model steps in a ML model step
n_block = 12  # number of ML model steps in a npz file
blocks_per_day = int(24 * 4 / (n_block * n_timestep_frequency))
n_blocks_train = int(30 * blocks_per_day)
n_blocks_window = int(7 * blocks_per_day)
n_blocks_val = n_blocks_window
