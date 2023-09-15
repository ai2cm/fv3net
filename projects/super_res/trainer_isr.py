import os

from model.isr_baseline import Trainer
from model.network_swinir import SwinIR
from data.load_data import load_data
from config_isr import config

def main():
    model = SwinIR(upscale=8, in_chans=12, out_chans=1, img_size=48, window_size=8,
        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6], embed_dim=200,
        num_heads=[8, 8, 8, 8, 8, 8, 8],
        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='3conv').cuda()

    train_dl, val_dl = load_data(
            config.data_config,
            config.batch_size,
            pin_memory = True,
            num_workers = 4,
        )

    trainer = Trainer(
        model,
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
        #tensorboard_dir = os.path.join(config.tensorboard_dir, f"{config.model_name}/"),
    )

    trainer.train()


if __name__ == "__main__":
    print(config)
    main()
