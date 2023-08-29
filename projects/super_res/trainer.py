import os

from model.autoreg_diffusion_mod import Unet, Flow, GaussianDiffusion, Trainer
from data.load_data import load_data
from config import config

def main():

    if config.data_config["multi"]:

        # in_ch_model = 2 * config.data_config["img_channel"] + 4 + 1 # all channels plus noise : (1 + 4 + 1) + 1 : (precip + multi + topo) + noise
        # in_ch_flow = 3 * (config.data_config["img_channel"] + 4 + 1) # all channels from current low res and past two high res : 3 * (1 + 4 + 1) : 3 * (precip + multi + topo)
        # in_ch_isr = config.data_config["img_channel"] + 4 + 1 # all channels from current low res : 1 + 4 + 1 : precip + multi + topo
        in_ch_model = 2 * config.data_config["img_channel"] + 10 + 1 # all channels plus noise : (1 + 4 + 1) + 1 : (precip + multi + topo) + noise
        in_ch_flow = 3 * (config.data_config["img_channel"] + 10 + 1) # all channels from current low res and past two high res : 3 * (1 + 4 + 1) : 3 * (precip + multi + topo)
        in_ch_isr = config.data_config["img_channel"] + 10 + 1 # all channels from current low res : 1 + 4 + 1 : precip + multi + topo

    else:

        in_ch_model = 2 * config.data_config["img_channel"]
        in_ch_flow = 3 * config.data_config["img_channel"]
        in_ch_isr = config.data_config["img_channel"]

    if config.data_config["flow"] == "3d":

        out_ch_flow = 3

    elif config.data_config["flow"] == "2d":

        out_ch_flow = 2

    model = Unet(
        dim = config.dim,
        channels = in_ch_model,
        out_dim = config.data_config["img_channel"],
        dim_mults = config.dim_mults,
        learned_sinusoidal_cond = config.learned_sinusoidal_cond,
        random_fourier_features = config.random_fourier_features,
        learned_sinusoidal_dim = config.learned_sinusoidal_dim
    ).cuda()

    flow = Flow(
        dim = config.dim,
        channels = in_ch_flow,
        out_dim = out_ch_flow,
        dim_mults = config.dim_mults
    ).cuda()
    
    diffusion = GaussianDiffusion(
        model,
        flow,
        image_size = config.data_config["img_size"],
        in_ch = in_ch_isr,
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
        eval_folder = os.path.join(config.eval_folder, f"{config.model_name}/"),
        results_folder = os.path.join(config.results_folder, f"{config.model_name}/"),
        config = config
    )

    trainer.train()


if __name__ == "__main__":
    print(config)
    main()