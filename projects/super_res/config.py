dim = 64
dim_mults = (1, 1, 2, 2, 4, 4)
learned_sinusoidal_cond = True,
random_fourier_features = True,
learned_sinusoidal_dim = 32
diffusion_steps = 1400
sampling_steps = 6
loss = "l1"
objective = "pred_v"
lr = 8e-5
steps = 5000000
grad_acc = 2
val_num_of_batch = 30
save_and_sample_every = 5000
ema_decay = 0.995
amp = False
split_batches = True
additional_note = ""
eval_folder = "./evaluate"
results_folder = "./results"
tensorboard_dir = "./tensorboard"
milestone = 1

batch_size = 4
data_config = {
    "dataset_name": "c384",
    "length": 7,
    "channels": ["CPRATEsfc_coarse","DLWRFsfc_coarse"],
    "img_channel": 2,
    "img_size": 384
}

data_name = f"{data_config['dataset_name']}-{data_config['channels']}-{objective}-{loss}-d{dim}-t{diffusion_steps}{additional_note}"
model_name = f"c384-{data_config['channels']}-{objective}-{loss}-d{dim}-t{diffusion_steps}{additional_note}"