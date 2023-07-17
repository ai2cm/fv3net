import os

from model.autoreg_diffusion import Unet, Flow, GaussianDiffusion, Trainer
from data.load_data import load_data
from config_infer import config

model = Unet(
    dim = config.dim,
    channels = 2 * config.data_config["img_channel"],
    out_dim = config.data_config["img_channel"],
    dim_mults = config.dim_mults,
    learned_sinusoidal_cond = config.learned_sinusoidal_cond,
    random_fourier_features = config.random_fourier_features,
    learned_sinusoidal_dim = config.learned_sinusoidal_dim
).cuda()

flow = Flow(
    dim = config.dim,
    channels = 3 * config.data_config["img_channel"],
    out_dim = 3,
    dim_mults = config.dim_mults
).cuda()

diffusion = GaussianDiffusion(
    model,
    flow,
    image_size = config.data_config["img_size"],
    timesteps = config.diffusion_steps,
    sampling_timesteps = config.sampling_steps,
    loss_type = config.loss,
    objective = config.objective
).cuda()

train_dl, val_dl = load_data(
        config.data_config,
        config.batch_size,
        pin_memory = True,
        num_workers = 4,
    )

trainer = Trainer(
    diffusion,
    train_dl,
    val_dl,
    train_batch_size = config.batch_size,
    train_lr = config.lr,
    train_num_steps = config.steps,
    gradient_accumulate_every = config.grad_acc,
    val_num_of_batch = config.val_num_of_batch,
    save_and_sample_every = config.save_and_sample_every,
    ema_decay = config.ema_decay,
    amp = config.amp,
    split_batches = config.split_batches,
    #eval_folder = os.path.join(config.eval_folder, f"{config.model_name}/"),
    eval_folder = os.path.join(config.eval_folder, f"{config.data_name}/"),
    results_folder = os.path.join(config.results_folder, f"{config.model_name}/"),
    config = config
    #tensorboard_dir = os.path.join(config.tensorboard_dir, f"{config.model_name}/"),
)

trainer.load(config.milestone)

trainer.sample()