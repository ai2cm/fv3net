from ml_collections import config_dict

config = config_dict.ConfigDict()

config.lr = 1e-4
config.steps = 700000
config.grad_acc = 1
config.val_num_of_batch = 5
config.save_and_sample_every = 20000
config.ema_decay = 0.995
config.amp = False
config.split_batches = True
config.additional_note = "isr"
config.eval_folder = "./evaluate"
config.results_folder = "./results"
config.tensorboard_dir = "./tensorboard"
config.milestone = 1
config.rollout = None
config.rollout_batch = None

config.batch_size = 1
config.data_config = config_dict.ConfigDict({
    "dataset_name": "c384",
    "length": 7,
    #"channels": ["UGRD10m_coarse","VGRD10m_coarse"],
    "channels": ["PRATEsfc_coarse"],
    #"img_channel": 2,
    "img_channel": 1,
    "img_size": 384,
    "logscale": True,
    "multi": True,
    "flow": "2d",
    "minipatch": False
})

config.data_name = f"{config.data_config['dataset_name']}-{config.data_config['channels']}-{config.additional_note}"
config.model_name = f"c384-{config.data_config['channels']}-{config.additional_note}"