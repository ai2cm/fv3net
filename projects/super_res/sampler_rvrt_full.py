import os
from torch import nn
from model.denoising_diffusion_rvrt_full import RSTBWithInputConv, Upsample, GuidedDeformAttnPack, GaussianDiffusion, SpyNet, Trainer
from config_rvrt_full_infer import config

recon = RSTBWithInputConv(
    in_channels = 5 * config.dim,
    kernel_size = (1, 3, 3),
    stride = 1,
    groups = 1,
    num_blocks = 1,
    dim = config.dim,
    input_resolution = config.data_config["img_size"],
    num_heads = 6,
    depth = 2,
    window_size = (1,8,8)
).cuda()

feat_ext = RSTBWithInputConv(
    in_channels = config.data_config["img_channel"]+11,
    kernel_size = (1, 3, 3),
    stride = 1,
    groups = 1,
    num_blocks = 1,
    dim = config.dim,
    input_resolution = config.data_config["img_size"],
    num_heads = 6,
    depth = 2,
    window_size = (1,8,8)
).cuda()

feat_up = Upsample(
    scale = 8,
    num_feat = config.dim,
    in_channels = config.data_config["img_channel"]
).cuda()

spynet = SpyNet('./spynet').cuda()

backbone = nn.ModuleDict()
deform_align = nn.ModuleDict()\

modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']

for i, module in enumerate(modules):
    # deformable attention
    deform_align[module] = GuidedDeformAttnPack(config.dim,
                                                config.dim,
                                                attention_window=[3, 3],
                                                attention_heads=6,
                                                deformable_groups=6,
                                                clip_size=2,
                                                max_residue_magnitude=10).cuda()

    # feature propagation
    backbone[module] = RSTBWithInputConv(
                                            in_channels = (2 + i) * config.dim,
                                            kernel_size = (1, 3, 3),
                                            stride = 1,
                                            groups = 1,
                                            num_blocks = 2,
                                            dim = config.dim,
                                            input_resolution = config.data_config["img_size"],
                                            num_heads = 6,
                                            depth = 2,
                                            window_size = (2,8,8)
                                        ).cuda()

diffusion = GaussianDiffusion(
    feat_ext = feat_ext,
    feat_up = feat_up,
    backbone = backbone,
    deform_align = deform_align,
    recon = recon,
    spynet = spynet,
    image_size = config.data_config["img_size"],
    timesteps = config.diffusion_steps,
    sampling_timesteps = config.sampling_steps,
    loss_type = config.loss,
    objective = config.objective
).cuda()

trainer = Trainer(
    diffusion,
    None,
    None,
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

trainer.load(config.milestone)

trainer.sample()