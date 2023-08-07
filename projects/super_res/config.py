from ml_collections import config_dict

#batch_size = 4
config = config_dict.ConfigDict()

config.dim = 64
config.dim_mults = (1, 1, 2, 2, 3, 4)
config.learned_sinusoidal_cond = True,
config.random_fourier_features = True,
config.learned_sinusoidal_dim = 32
config.diffusion_steps = 1500
config.sampling_steps = 20
config.loss = "l2"
config.objective = "pred_v"
config.lr = 8e-5
config.steps = 5000000
config.grad_acc = 1
config.val_num_of_batch = 2
config.save_and_sample_every = 5000
config.ema_decay = 0.995
config.amp = False
config.split_batches = True
config.additional_note = "no-logscale"
config.eval_folder = "./evaluate"
config.results_folder = "./results"
config.tensorboard_dir = "./tensorboard"
config.milestone = 1

config.batch_size = 1
config.data_config = config_dict.ConfigDict({
    "dataset_name": "c384",
    "length": 7,
    #"channels": ["UGRD10m_coarse","VGRD10m_coarse"],
    "channels": ["PRATEsfc_coarse"],
    #"img_channel": 2,
    "img_channel": 1,
    "img_size": 384,
    "logscale": False,
    "quick": True
})

config.data_name = f"{config.data_config['dataset_name']}-{config.data_config['channels']}-{config.objective}-{config.loss}-d{config.dim}-t{config.diffusion_steps}{config.additional_note}"
config.model_name = f"c384-{config.data_config['channels']}-{config.objective}-{config.loss}-d{config.dim}-t{config.diffusion_steps}{config.additional_note}"