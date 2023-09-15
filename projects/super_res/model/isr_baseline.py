from pathlib import Path
import os

import numpy as np
import xarray as xr

import torch
import wandb

import piq
import pickle
import cv2
from scipy.stats import wasserstein_distance

from torch.optim import Adam
import torch.nn.functional as F

from random import randint
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm.auto import tqdm
from ema_pytorch import EMA
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable as smap

from accelerate import Accelerator
from einops import rearrange, reduce

def get_random_idx_with_difference(min_tx, max_tx, number_tx, diff):
    times = []
    while len(times) < number_tx:
        new_time = randint(min_tx, max_tx)
        if all(abs(new_time - time) >= diff for time in times):
            times.append(new_time)
    return times

def cycle(dl):
    while True:
        for data in dl:
            yield data

def exists(x):
    return x is not None

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# trainer class

class Trainer(object):
    def __init__(
        self,
        model,
        train_dl,
        val_dl,
        config,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        #augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 1,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 10,
        #num_samples = 25,
        eval_folder = './evaluate',
        results_folder = './results',
        #tensorboard_dir = './tensorboard',
        val_num_of_batch = 2,
        amp = False,
        fp16 = False,
        #fp16 = True,
        split_batches = True,
        #split_batches = False,
        convert_image_to = None
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no',
            log_with = 'wandb',
        )
        self.accelerator.init_trackers("climate", 
            init_kwargs={
                "wandb": {
                    "name":  None,
                }
            },
        )
        self.config = config
        self.accelerator.native_amp = amp
        self.multi = config.data_config["multi"]
        self.rollout = config.rollout
        self.rollout_batch = config.rollout_batch
        self.flow = config.data_config["flow"]
        self.minipatch = config.data_config["minipatch"]
        self.logscale = config.data_config["logscale"]

        self.model = model

        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        self.val_num_of_batch = val_num_of_batch
        
        # optimizer

        self.opt = Adam(model.parameters(), lr = train_lr, betas = adam_betas)
        self.sched = CosineAnnealingLR(self.opt, train_num_steps, 5e-7)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(model, beta = ema_decay, update_every = ema_update_every)
            
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(exist_ok=True, parents=True)
        
        self.eval_folder = Path(eval_folder)

        self.eval_folder.mkdir(exist_ok=True, parents=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, train_dl, val_dl = self.accelerator.prepare(self.model, self.opt, train_dl, val_dl)
        self.train_dl = cycle(train_dl)
        self.val_dl = val_dl

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            #'version': __version__
        }

        torch.save(data, str(self.results_folder / f'qmodel-{milestone%3}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'qmodel-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        #self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        #if 'version' in data:
        #    print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):

        accelerator = self.accelerator
        device = accelerator.device

        cmap = mpl.colormaps['RdBu_r']
        fcmap = mpl.colormaps['gray_r']

        with open("data/ensemble_c48_trainstats/chl.pkl", 'rb') as f:
            c48_chl = pickle.load(f)
        
        with open("data/ensemble_c48_trainstats/log_chl.pkl", 'rb') as f:
            c48_log_chl = pickle.load(f)

        with open("data/ensemble_c384_trainstats/chl.pkl", 'rb') as f:
            c384_chl = pickle.load(f)

        with open("data/ensemble_c384_trainstats/log_chl.pkl", 'rb') as f:
            c384_log_chl = pickle.load(f)

        c384_lgmin = c384_log_chl["PRATEsfc"]['min']
        c384_lgmax = c384_log_chl["PRATEsfc"]['max']
        c48_lgmin = c48_log_chl["PRATEsfc_coarse"]['min']
        c48_lgmax = c48_log_chl["PRATEsfc_coarse"]['max']
        
        c384_min = c384_chl["PRATEsfc"]['min']
        c384_max = c384_chl["PRATEsfc"]['max']
        c48_min = c48_chl["PRATEsfc_coarse"]['min']
        c48_max = c48_chl["PRATEsfc_coarse"]['max']

        c384_gmin = c384_min
        c48_gmin = c48_min

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    
                    data = next(self.train_dl)
                    lres = data['LR'].to(device)
                    hres = data['HR'].to(device)

                    with self.accelerator.autocast():

                        topo = hres[:, :, 1:2, :, :]
                        hres = hres[:, :, 0:1, :, :]
                        topo_low = rearrange(F.interpolate(rearrange(topo, 'b t c h w -> (b t) c h w'), size=(48, 48), mode='bilinear'), '(b t) c h w -> b t c h w', t = 7)
                        
                        ures = self.model(rearrange(torch.cat((lres, topo_low), dim = 2), 'b t c h w -> (b t) c h w'))
                        loss = F.mse_loss(ures, rearrange(hres, 'b t c h w -> (b t) c h w'), reduction = 'none')
                        loss = reduce(loss, 'b ... -> b (...)', 'mean')
                        loss = loss.mean()

                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.log({"loss": total_loss}, step = self.step)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():

                            vlosses = []
                            vids = []
                            hr = []
                            lr = []
                            num_frames = 5
                            img_size = 384
                            
                            for i, batch in enumerate(self.val_dl):
                                
                                lres = batch['LR'].to(device)
                                hres = batch['HR'].to(device)
                                
                                if i >= self.val_num_of_batch:
                                    break
                                
                                topo = hres[:, :, 1:2, :, :]
                                hres = hres[:, :, 0:1, :, :]
                                topo_low = rearrange(F.interpolate(rearrange(topo, 'b t c h w -> (b t) c h w'), size=(48, 48), mode='bilinear'), '(b t) c h w -> b t c h w', t = 7)
                                
                                ures = self.model(rearrange(torch.cat((lres, topo_low), dim = 2), 'b t c h w -> (b t) c h w'))
                                loss = F.mse_loss(ures, rearrange(hres, 'b t c h w -> (b t) c h w'), reduction = 'none')
                                
                                videos = rearrange(ures, '(b t) c h w -> b t c h w', t = 7)

                                vids.append(videos)
                                vlosses.append(loss)
                                hr.append(hres)
                                lr.append(lres)

                            videos = torch.cat(vids, dim = 0)
                            vloss = torch.stack(vlosses, dim = 0).mean()
                            #self.sched.step(vloss)
                            hres = torch.cat(hr, dim = 0)
                            lres = torch.cat(lr, dim = 0)
                            del vids, vlosses, hr, lr

                            lres = lres[:, :, 0:1, :, :]
                            hres = hres[:, :, 0:1, :, :]

                            if not self.logscale:
                                target = hres[:,:,:,:,:].detach().cpu().numpy() * (c384_max - c384_min) + c384_min
                                output = videos.detach().cpu().numpy() * (c384_max - c384_min) + c384_min
                                coarse = lres[:,:,:,:,:].detach().cpu().numpy() * (c48_max - c48_min) + c48_min
                            
                            else:
                                target = hres[:,:,:,:,:].detach().cpu().numpy() * (c384_lgmax - c384_lgmin) + c384_lgmin
                                output = videos.detach().cpu().numpy() * (c384_lgmax - c384_lgmin) + c384_lgmin
                                coarse = lres[:,:,:,:,:].detach().cpu().numpy() * (c48_lgmax - c48_lgmin) + c48_lgmin
                            
                            if self.logscale:
                                target = np.exp(target) + c384_gmin - 1e-14
                                output = np.exp(output) + c384_gmin - 1e-14
                                coarse = np.exp(coarse) + c48_gmin - 1e-14

                            ssim_index = piq.ssim(torch.from_numpy(target).view(-1, 1, 384, 384), torch.from_numpy(output).view(-1, 1, 384, 384).clamp(0., 1.), data_range=1., reduction='none')
                            gmsd_index = piq.gmsd(torch.from_numpy(target).view(-1, 1, 384, 384), torch.from_numpy(output).view(-1, 1, 384, 384).clamp(0., 1.), data_range=1., reduction='none')

                            nn_upscale = np.repeat(np.repeat(coarse, 8, axis = 3), 8, axis = 4)
                            diff_output = (output - nn_upscale).flatten()
                            diff_target = (target - nn_upscale).flatten()
                            vmin = min(diff_output.min(), diff_target.min())
                            vmax = max(diff_output.max(), diff_target.max())
                            bins = np.linspace(vmin, vmax, 100 + 1)

                            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                            ax.hist(
                                diff_output, bins=bins, alpha=0.5, label="Output", histtype="step", density=True
                            )
                            ax.hist(
                                diff_target, bins=bins, alpha=0.5, label="Target", histtype="step", density=True
                            )
                            ax.set_xlim(vmin, vmax)
                            ax.legend()
                            ax.set_ylabel("Density")
                            ax.set_yscale("log")

                            output1 = output.flatten()
                            target1 = target.flatten()
                            rmse = np.sqrt(np.mean((output1 - target1)**2))
                            pscore = np.abs(np.percentile(output1, 99.999) - np.percentile(target1, 99.999))
                            vmin1 = min(output1.min(), target1.min())
                            vmax1 = max(output1.max(), target1.max())
                            bins1 = np.linspace(vmin1, vmax1, 100 + 1)
                            #histo = np.histogram(output1, bins=bins1, density=True)[0].ravel().astype('float32')
                            #histt = np.histogram(target1, bins=bins1, density=True)[0].ravel().astype('float32')
                            count_o, bin_o = np.histogram(output1, bins=bins1, density=True)
                            count_t, bin_t = np.histogram(target1, bins=bins1, density=True)
                            histo = count_o.ravel().astype('float32')
                            histt = count_t.ravel().astype('float32')
                            distchisqr = cv2.compareHist(histo, histt, cv2.HISTCMP_CHISQR)
                            distinter = cv2.compareHist(histo, histt, cv2.HISTCMP_INTERSECT)
                            distkl = cv2.compareHist(histo, histt, cv2.HISTCMP_KL_DIV)
                            distemd = wasserstein_distance(output1, target1)                            

                            fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
                            ax1.hist(
                                #output1, bins=bins1, alpha=0.5, label="Output", histtype="step", density=True
                                bin_o[:-1], bins=bin_o, weights = count_o, alpha=0.5, label="Output", histtype="step"#, density=True
                            )
                            ax1.hist(
                                #target1, bins=bins1, alpha=0.5, label="Target", histtype="step", density=True
                                bin_t[:-1], bins=bin_t, weights = count_t, alpha=0.5, label="Target", histtype="step"#, density=True
                            )
                            ax1.set_xlim(vmin1, vmax1)
                            ax1.legend()
                            ax1.set_ylabel("Density")
                            ax1.set_yscale("log")
                            
                            if self.logscale:
                                
                                accelerator.log({"true_high": wandb.Video((hres[0:1,2:,0:1,:,:].repeat(1,1,3,1,1).cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"true_low": wandb.Video((lres[0:1,2:,0:1,:,:].repeat(1,1,3,1,1).cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"samples": wandb.Video((videos.clamp(0.0, 1.0)[0:1,:,0:1,:,:].repeat(1,1,3,1,1).detach().cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                            
                            else:

                                accelerator.log({"true_high": wandb.Video((hres[0:1,2:,0:1,:,:].repeat(1,1,3,1,1).cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"true_low": wandb.Video((lres[0:1,2:,0:1,:,:].repeat(1,1,3,1,1).cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"samples": wandb.Video((videos[0:1,:,:,:,:].clamp(0.0, 1.0).repeat(1,1,3,1,1).detach().cpu().numpy()*255).astype(np.uint8))}, step=self.step)
                                target = np.log(target - c384_gmin + 1e-14)
                                output = np.log(output - c384_gmin + 1e-14)
                                coarse = np.log(coarse - c48_gmin + 1e-14)
                                target = (target - c384_lgmin) / (c384_lgmax - c384_lgmin)
                                output = (output - c384_lgmin) / (c384_lgmax - c384_lgmin)
                                coarse = (coarse - c48_lgmin) / (c48_lgmax - c48_lgmin)
                                accelerator.log({"true_loghigh": wandb.Video((np.repeat(target[0:1,:,:,:,:], 3, axis=-3)*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"true_loglow": wandb.Video((np.repeat(coarse[0:1,:,:,:,:], 3, axis=-3)*255).astype(np.uint8))}, step=self.step)
                                accelerator.log({"logsamples": wandb.Video((np.repeat(output[0:1,:,:,:,:], 3, axis=-3)*255).astype(np.uint8))}, step=self.step)

                            accelerator.log({"difference_histogram": wandb.Image(fig, mode = 'RGB')}, step=self.step)
                            accelerator.log({"histogram": wandb.Image(fig1, mode = 'RGB')}, step=self.step)
                            accelerator.log({"ssim": ssim_index.mean()}, step=self.step)
                            accelerator.log({"gmsd": gmsd_index.mean()}, step=self.step)
                            accelerator.log({"rmse": rmse}, step=self.step)
                            accelerator.log({"pscore": pscore}, step=self.step)
                            accelerator.log({"distchisqr": distchisqr}, step=self.step)
                            accelerator.log({"distinter": distinter}, step=self.step)
                            accelerator.log({"distkl": distkl}, step=self.step)
                            accelerator.log({"distemd": distemd}, step=self.step)
                            accelerator.log({"vloss": vloss}, step=self.step)
                            accelerator.log({"lr": self.opt.param_groups[0]['lr']}, step=self.step)
                                
                            milestone = self.step // self.save_and_sample_every
                            
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')

    def sample(self):

        accelerator = self.accelerator
        device = accelerator.device
        
        self.ema.ema_model.eval()

        PATH = "/extra/ucibdl0/shared/data/fv3gfs"
        XX = xr.open_zarr(f"{PATH}/c48_precip_plus_more_ave/0011/sfc_8xdaily_ave_coarse.zarr")
        XX_ = xr.open_zarr(f"{PATH}/c48_atmos_ave/0011/atmos_8xdaily_ave_coarse.zarr")
        yy = xr.open_zarr(f"{PATH}/c384_precip_ave/0011/sfc_8xdaily_ave.zarr")
        topot = xr.open_zarr(f"{PATH}/c384_topo/0011/atmos_static.zarr")

        with open("data/ensemble_c48_trainstats/chl.pkl", 'rb') as f:

            c48_chl = pickle.load(f)
        
        with open("data/ensemble_c48_trainstats/atm_chl.pkl", 'rb') as f:
            
            c48_atm_chl = pickle.load(f)

        with open("data/ensemble_c48_trainstats/log_chl.pkl", 'rb') as f:
            
            c48_log_chl = pickle.load(f)

        with open("data/ensemble_c384_trainstats/chl.pkl", 'rb') as f:

            c384_chl = pickle.load(f)

        with open("data/ensemble_c384_trainstats/log_chl.pkl", 'rb') as f:

            c384_log_chl = pickle.load(f)

        with open("data/ensemble_c384_trainstats/topo.pkl", 'rb') as f:

            c384_topo = pickle.load(f)

        if self.multi:

            c48_channels = ["PRATEsfc_coarse", "UGRD10m_coarse", "VGRD10m_coarse", "TMPsfc_coarse", "CPRATsfc_coarse", "DSWRFtoa_coarse"]
            c48_channels_atmos = ["ps_coarse", "u700_coarse", "v700_coarse", "vertically_integrated_liq_wat_coarse", "vertically_integrated_sphum_coarse"]
            c384_channels = ["PRATEsfc"]

        else:

            c48_channels = ["PRATEsfc_coarse"]
            c384_channels = ["PRATEsfc"]

        with torch.no_grad():

            for tile in range(6):

                if self.rollout == 'full':

                    seq_len = self.rollout_batch
                    st = 0
                    en = seq_len + 2
                    count = 0

                    while en < 3176:

                        print(tile, st)

                        X = XX.isel(time = slice(st, en), tile = tile)
                        X_ = XX_.isel(time = slice(st, en), tile = tile)
                        y = yy.isel(time = slice(st, en), tile = tile)


                        X = np.stack([X[channel].values for channel in c48_channels], axis = 1)
                        X_ = np.stack([X_[channel].values for channel in c48_channels_atmos], axis = 1)
                        y = np.stack([y[channel].values for channel in c384_channels], axis = 1)
                        topo = topot.isel(tile = tile)
                        topo = topo['zsurf'].values
                        topo = np.repeat(topo.reshape((1,1,384,384)), seq_len + 2, axis = 0)
                        
                        X[:,0:1,:,:] = np.log(X[:,0:1,:,:] - c48_chl["PRATEsfc_coarse"]['min'] + 1e-14)
                        y = np.log(y - c384_chl["PRATEsfc"]['min'] + 1e-14)
                        X[:,0:1,:,:] = (X[:,0:1,:,:] - c48_log_chl["PRATEsfc_coarse"]['min']) / (c48_log_chl["PRATEsfc_coarse"]['max'] - c48_log_chl["PRATEsfc_coarse"]['min'])
                        y = (y - c384_log_chl["PRATEsfc"]['min']) / (c384_log_chl["PRATEsfc"]['max'] - c384_log_chl["PRATEsfc"]['min'])

                        for i in range(1, X.shape[1]):

                            X[:,i,:,:] = (X[:,i,:,:] - c48_chl[c48_channels[i]]['min']) / (c48_chl[c48_channels[i]]['max'] - c48_chl[c48_channels[i]]['min'])

                        for i in range(X_.shape[1]):

                            X_[:,i,:,:] = (X_[:,i,:,:] - c48_atm_chl[c48_channels_atmos[i]]['min']) / (c48_atm_chl[c48_channels_atmos[i]]['max'] - c48_atm_chl[c48_channels_atmos[i]]['min'])

                        topo = (topo - c384_topo["zsurf"]['min']) / (c384_topo["zsurf"]['max'] - c384_topo["zsurf"]['min'])

                        X = np.concatenate((X, X_), axis = 1)
                        y = np.concatenate((y, topo), axis = 1)

                        lres = torch.from_numpy(X).unsqueeze(0).to(device)
                        hres = torch.from_numpy(y).unsqueeze(0).to(device)
                        topo = hres[:, :, 1:2, :, :]
                        topo_low = rearrange(F.interpolate(rearrange(topo, 'b t c h w -> (b t) c h w'), size=(48, 48), mode='bilinear'), '(b t) c h w -> b t c h w', t = seq_len + 2)

                        videos = self.model(rearrange(torch.cat((lres, topo_low), dim = 2), 'b t c h w -> (b t) c h w')).unsqueeze(0)

                        torch.save(videos, os.path.join(self.eval_folder) + "/gen_{}_{}.pt".format(tile, count))
                        torch.save(hres[:,:,0:1,:,:], os.path.join(self.eval_folder) + "/truth_hr_{}_{}.pt".format(tile, count))
                        torch.save(lres[:,:,0:1,:,:], os.path.join(self.eval_folder) + "/truth_lr_{}_{}.pt".format(tile, count))
                        count += 1

                        st += seq_len
                        en += seq_len

                if self.rollout == 'partial':

                    seq_len = self.rollout_batch
                    #indices = get_random_idx_with_difference(0, 3176 - (seq_len + 2), 75 // seq_len, seq_len + 2) # 75 samples per tile
                    indices = list(range(0, 3176 - (seq_len + 2), 250)) # deterministic, 325 samples per tile for seq_len of 25

                    for count, st in enumerate(indices):

                        print(tile, count)

                        X = XX.isel(time = slice(st, st+(seq_len+2)), tile = tile)
                        X_ = XX_.isel(time = slice(st, st+(seq_len+2)), tile = tile)
                        y = yy.isel(time = slice(st, st+(seq_len+2)), tile = tile)

                        X = np.stack([X[channel].values for channel in c48_channels], axis = 1)
                        X_ = np.stack([X_[channel].values for channel in c48_channels_atmos], axis = 1)
                        y = np.stack([y[channel].values for channel in c384_channels], axis = 1)
                        topo = topot.isel(tile = tile)
                        topo = topo['zsurf'].values
                        topo = np.repeat(topo.reshape((1,1,384,384)), seq_len + 2, axis = 0)

                        X[:,0:1,:,:] = np.log(X[:,0:1,:,:] - c48_chl["PRATEsfc_coarse"]['min'] + 1e-14)
                        y = np.log(y - c384_chl["PRATEsfc"]['min'] + 1e-14)
                        X[:,0:1,:,:] = (X[:,0:1,:,:] - c48_log_chl["PRATEsfc_coarse"]['min']) / (c48_log_chl["PRATEsfc_coarse"]['max'] - c48_log_chl["PRATEsfc_coarse"]['min'])
                        y = (y - c384_log_chl["PRATEsfc"]['min']) / (c384_log_chl["PRATEsfc"]['max'] - c384_log_chl["PRATEsfc"]['min'])

                        for i in range(1, X.shape[1]):

                            X[:,i,:,:] = (X[:,i,:,:] - c48_chl[c48_channels[i]]['min']) / (c48_chl[c48_channels[i]]['max'] - c48_chl[c48_channels[i]]['min'])

                        for i in range(X_.shape[1]):

                            X_[:,i,:,:] = (X_[:,i,:,:] - c48_atm_chl[c48_channels_atmos[i]]['min']) / (c48_atm_chl[c48_channels_atmos[i]]['max'] - c48_atm_chl[c48_channels_atmos[i]]['min'])

                        topo = (topo - c384_topo["zsurf"]['min']) / (c384_topo["zsurf"]['max'] - c384_topo["zsurf"]['min'])

                        X = np.concatenate((X, X_), axis = 1)
                        y = np.concatenate((y, topo), axis = 1)

                        lres = torch.from_numpy(X).unsqueeze(0).to(device)
                        hres = torch.from_numpy(y).unsqueeze(0).to(device)
                        topo = hres[:, :, 1:2, :, :]
                        topo_low = rearrange(F.interpolate(rearrange(topo, 'b t c h w -> (b t) c h w'), size=(48, 48), mode='bilinear'), '(b t) c h w -> b t c h w', t = seq_len + 2)

                        videos = self.model(rearrange(torch.cat((lres, topo_low), dim = 2), 'b t c h w -> (b t) c h w')).unsqueeze(0)

                        torch.save(videos, os.path.join(self.eval_folder) + "/gen_{}_{}.pt".format(tile, count))
                        torch.save(hres[:,:,0:1,:,:], os.path.join(self.eval_folder) + "/truth_hr_{}_{}.pt".format(tile, count))
                        torch.save(lres[:,:,0:1,:,:], os.path.join(self.eval_folder) + "/truth_lr_{}_{}.pt".format(tile, count))