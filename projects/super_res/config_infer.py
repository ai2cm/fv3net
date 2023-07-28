from ml_collections import config_dict

config = config_dict.ConfigDict()


config.dim = 64
config.dim_mults = (1, 1, 2, 2, 4, 4)
config.learned_sinusoidal_cond = True,
config.random_fourier_features = True,
config.learned_sinusoidal_dim = 32
config.diffusion_steps = 1500
config.sampling_steps = 6
config.loss = "l1"
config.objective = "pred_v"
config.lr = 8e-5
config.steps = 5000000
config.grad_acc = 2
config.val_num_of_batch = 5
config.save_and_sample_every = 5000
config.ema_decay = 0.995
config.amp = False
config.split_batches = True
config.additional_note = ""
config.eval_folder = "./evaluate"
config.results_folder = "./results"
config.tensorboard_dir = "./tensorboard"
config.milestone = 1

config.batch_size = 4
config.data_config = config_dict.ConfigDict({
    "dataset_name": "c384",
    "length": 7,
    #"channels": ["UGRD10m_coarse","VGRD10m_coarse"],
    "channels": ["PRATEsfc_coarse"],
    #"img_channel": 2,
    "img_channel": 1,
    "img_size": 384,
    "logscale": True
})

data_name = f"{config.data_config['dataset_name']}-{config.data_config['channels']}-{config.objective}-{config.loss}-d{config.dim}-t{config.diffusion_steps}{config.additional_note}"
model_name = f"c384-{config.data_config['channels']}-{config.objective}-{config.loss}-d{config.dim}-t{config.diffusion_steps}{config.additional_note}"