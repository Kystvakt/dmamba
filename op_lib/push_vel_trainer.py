import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from .dist_util import local_rank, is_leader_process
from .downsample import downsample_domain
from .losses import LpLoss
from .metrics import compute_metrics, write_metrics
from .plt_util import plt_temp, plt_iter_mae, plt_vel

t_bulk_map = {
    'wall_super_heat': 58,
    'subcooled': 50
}


class PushVelTrainer:
    def __init__(self,
                 model,
                 future_window,
                 push_forward_steps,
                 train_dataloader,
                 val_dataloader,
                 optimizer,
                 lr_scheduler,
                 val_variable,
                 writer,
                 cfg):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.val_variable = val_variable
        self.writer = writer
        self.cfg = cfg
        self.device = self.cfg.model.device
        self.log_to_wandb = self.cfg.log_to_wandb
        if self.log_to_wandb:
            self.wandb_config = self.cfg.wandb
        self.loss = LpLoss(d=2, reduce_dims=[0, 1])
        self.push_forward_steps = push_forward_steps
        self.future_window = future_window
        self.local_rank = local_rank()

    def save_checkpoint(self, log_dir, epoch):
        if self.cfg.distributed:
            model_name = self.model.module.__class__.__name__.lower()
        else:
            model_name = self.model.__class__.__name__.lower()
        ckpt_file = f'{model_name}_{self.cfg.torch_dataset_name.split("_")[0]}_{self.cfg.exp_num}_{epoch}.pt'
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ckpt_path = f'{log_dir}/{ckpt_file}'
        print(f'saving model to {ckpt_path}')
        if self.cfg.distributed:
            torch.save(self.model.module.state_dict(), f'{ckpt_path}')
        else:
            torch.save(self.model.state_dict(), f'{ckpt_path}')
        return ckpt_path

    def push_forward_prob(self, epoch, max_epochs):
        r"""
        Randomly set the number of push-forward steps based on current
        iteration. Initially, it's unlike to "push-forward." later in training,
        it's nearly certain to apply the push-forward trick.
        """
        cur_iter = epoch * len(self.train_dataloader)
        tot_iter = max_epochs * len(self.train_dataloader)
        frac = cur_iter / tot_iter
        if np.random.uniform() > frac:
            return 1
        else:
            return self.push_forward_steps

    def train(self, max_epochs, log_dir, dataset_name):
        best_rmse = 1e+5
        save_metrics = ({}, {}, {})

        for epoch in range(max_epochs):
            print('epoch ', epoch)
            self.train_step(epoch, max_epochs)
            self.val_step(epoch)
            val_dataset = self.val_dataloader.dataset.datasets[0]
            temp_metrics, velx_metrics, vely_metrics = self.test(val_dataset, log_dir)
            
            # Save model based on the average rmse
            avg_rmse = (temp_metrics.rmse + velx_metrics.rmse + vely_metrics.rmse) / 3
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                save_metrics = (temp_metrics, velx_metrics, vely_metrics)
                ckpt_path = self.save_checkpoint(log_dir, epoch)

            if self.log_to_wandb:
                test_logs = defaultdict(float)
                test_logs['Temp_MAE'] = temp_metrics.mae
                test_logs['Temp_RMSE'] = temp_metrics.rmse
                test_logs['Temp_Relative Error'] = temp_metrics.relative_error
                test_logs['Temp_Max Error'] = temp_metrics.max_error
                test_logs['Temp_Boundary RMSE'] = temp_metrics.boundary_rmse
                test_logs['Temp_Interface RMSE'] = temp_metrics.interface_rmse
                test_logs['Temp_Fourier - Low'] = temp_metrics.fourier_low
                test_logs['Temp_Fourier - Mid'] = temp_metrics.fourier_mid
                test_logs['Temp_Fourier - High'] = temp_metrics.fourier_high

                test_logs['velx_MAE'] = velx_metrics.mae
                test_logs['velx_RMSE'] = velx_metrics.rmse
                test_logs['velx_Relative Error'] = velx_metrics.relative_error
                test_logs['velx_Max Error'] = velx_metrics.max_error
                test_logs['velx_Boundary RMSE'] = velx_metrics.boundary_rmse
                test_logs['velx_Interface RMSE'] = velx_metrics.interface_rmse
                test_logs['velx_Fourier - Low'] = velx_metrics.fourier_low
                test_logs['velx_Fourier - Mid'] = velx_metrics.fourier_mid
                test_logs['velx_Fourier - High'] = velx_metrics.fourier_high

                test_logs['vely_MAE'] = vely_metrics.mae
                test_logs['vely_RMSE'] = vely_metrics.rmse
                test_logs['vely_Relative Error'] = vely_metrics.relative_error
                test_logs['vely_Max Error'] = vely_metrics.max_error
                test_logs['vely_Boundary RMSE'] = vely_metrics.boundary_rmse
                test_logs['vely_Interface RMSE'] = vely_metrics.interface_rmse
                test_logs['vely_Fourier - Low'] = vely_metrics.fourier_low
                test_logs['vely_Fourier - Mid'] = vely_metrics.fourier_mid
                test_logs['vely_Fourier - High'] = vely_metrics.fourier_high
                wandb.log(test_logs, step=epoch)
            
        # Print metrics of best model after training is finished.
        print(save_metrics)

        return ckpt_path

    def _forward_int(self, coords, temp, vel, dfun):
        # TODO: account for possibly different timestep sizes of training data
        input = torch.cat((temp, vel, dfun), dim=1)
        if self.cfg.train.use_coords:
            input = torch.cat((coords, input), dim=1)
        pred = self.model(input)

        # timesteps = (torch.arange(self.future_window) + 1).cuda().unsqueeze(-1).unsqueeze(-1).float()
        # timesteps /= 10 # timestep size is 0.1 for vel
        # timesteps = timesteps.to(pred.device)

        # d_temp = pred[:, :self.future_window]
        # last_temp_input = temp[:, -1].unsqueeze(1)
        # temp_pred = last_temp_input + timesteps * d_temp

        # d_vel = pred[:, self.future_window:]
        # last_vel_input = vel[:, -2:].repeat(1, self.future_window, 1, 1)
        # timesteps_interleave = torch.repeat_interleave(timesteps, 2, dim=0)
        # vel_pred = last_vel_input + timesteps_interleave * d_vel

        temp_pred = pred[:, :self.future_window]
        vel_pred = pred[:, self.future_window:]

        return temp_pred, vel_pred

    def _index_push(self, idx, coords, temp, vel, dfun):
        r"""
        select the channels for push_forward_step `idx`
        """
        return (coords[:, idx], temp[:, idx], vel[:, idx], dfun[:, idx])

    def _index_dfun(self, idx, dfun):
        return dfun[:, idx]

    def push_forward_trick(self, coords, temp, vel, dfun, push_forward_steps):
        # TODO: clean this up...
        coords_input, temp_input, vel_input, dfun_input = self._index_push(0, coords, temp, vel, dfun)
        assert self.future_window == temp_input.size(1), 'push-forward expects history size to match future'
        coords_input, temp_input, vel_input, dfun_input = \
            downsample_domain(self.cfg.train.downsample_factor, coords_input, temp_input, vel_input, dfun_input)
        with torch.no_grad():
            for idx in range(push_forward_steps - 1):
                temp_input, vel_input = self._forward_int(coords_input, temp_input, vel_input, dfun_input)
                dfun_input = self._index_dfun(idx + 1, dfun)
                dfun_input = downsample_domain(self.cfg.train.downsample_factor, dfun_input)[0]
        if self.cfg.train.noise and push_forward_steps == 1:
            temp_input += torch.empty_like(temp_input).normal_(0, 0.01)
            vel_input += torch.empty_like(vel_input).normal_(0, 0.01)
        temp_pred, vel_pred = self._forward_int(coords_input, temp_input, vel_input, dfun_input)
        return temp_pred, vel_pred

    def train_step(self, epoch, max_epochs):
        self.model.train()
        epoch_temp_loss = 0
        epoch_vel_loss = 0
        epoch_loss = 0
        start = time.time()
        # warmup before doing push forward trick
        for iter, (coords, temp, vel, dfun, temp_label, vel_label) in enumerate(self.train_dataloader):
            coords = coords.to(self.device).float()
            temp = temp.to(self.device).float()
            vel = vel.to(self.device).float()
            dfun = dfun.to(self.device).float()
            
            push_forward_steps = self.push_forward_prob(epoch, max_epochs)

            temp_pred, vel_pred = self.push_forward_trick(coords, temp, vel, dfun, push_forward_steps)
            idx = (push_forward_steps - 1)
            temp_label = temp_label[:, idx].to(self.device).float()
            vel_label = vel_label[:, idx].to(self.device).float()
            temp_label, vel_label = downsample_domain(self.cfg.train.downsample_factor, temp_label, vel_label)
            
            temp_loss = F.mse_loss(temp_pred, temp_label)
            vel_loss = F.mse_loss(vel_pred, vel_label)
            loss = (temp_loss + vel_loss) / 2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            epoch_temp_loss += temp_loss
            epoch_vel_loss += vel_loss
            epoch_loss += loss
            global_iter = epoch * len(self.train_dataloader) + iter
            write_metrics(temp_pred, temp_label, global_iter, 'TrainTemp', self.writer)
            write_metrics(vel_pred, vel_label, global_iter, 'TrainVel', self.writer)
            del temp, vel, temp_label, vel_label

        end = time.time()
        print(
            f'Epoch : {epoch} | Avg MSE: {epoch_loss / len(self.train_dataloader):.4f} | Temp MSE: {epoch_temp_loss / len(self.train_dataloader):.4f} | Vel MSE: {epoch_vel_loss / len(self.train_dataloader):.4f} | Time: {end - start:.2f}s')
        if self.log_to_wandb:
            train_logs = defaultdict(float)
            train_logs['train_loss'] = epoch_loss / len(self.train_dataloader)
            train_logs['train_temp_loss'] = epoch_temp_loss / len(self.train_dataloader)
            train_logs['train_vel_loss'] = epoch_vel_loss / len(self.train_dataloader)
            wandb.log(train_logs, step=epoch)

    def val_step(self, epoch):
        val_epoch_loss = 0
        val_temp_loss = 0
        val_vel_loss = 0
        self.model.eval()
        start = time.time()
        for iter, (coords, temp, vel, dfun, temp_label, vel_label) in enumerate(self.val_dataloader):
            coords = coords.to(self.device).float()
            temp = temp.to(self.device).float()
            vel = vel.to(self.device).float()
            dfun = dfun.to(self.device).float()

            # val doesn't apply push-forward
            temp_label = temp_label[:, 0].to(self.device).float()
            vel_label = vel_label[:, 0].to(self.device).float()

            with torch.no_grad():
                temp_pred, vel_pred = self._forward_int(coords[:, 0], temp[:, 0], vel[:, 0], dfun[:, 0])
                temp_loss = F.mse_loss(temp_pred, temp_label)
                vel_loss = F.mse_loss(vel_pred, vel_label)
                loss = (temp_loss + vel_loss) / 2
        
            val_epoch_loss += loss
            val_temp_loss += temp_loss
            val_vel_loss += vel_loss
            global_iter = epoch * len(self.val_dataloader) + iter
            write_metrics(temp_pred, temp_label, global_iter, 'ValTemp', self.writer)
            write_metrics(vel_pred, vel_label, global_iter, 'ValVel', self.writer)
            del temp, vel, temp_label, vel_label

        end = time.time()
        print(
            f'Epoch : {epoch} | Avg Val MSE: {val_epoch_loss / len(self.val_dataloader):.4f} | Val Temp MSE: {val_temp_loss / len(self.val_dataloader):.4f} | Val Vel MSE: {val_vel_loss / len(self.val_dataloader):.4f} | Time: {end - start :.2f}s')
        if self.log_to_wandb:
            val_logs = defaultdict(float)
            val_logs['val_loss'] = val_epoch_loss / len(self.val_dataloader)
            val_logs['val_temp_loss'] = val_temp_loss / len(self.val_dataloader)
            val_logs['val_vel_loss'] = val_vel_loss / len(self.val_dataloader)
            wandb.log(val_logs, step=epoch)

    def test(self, dataset, log_dir, max_time_limit=200):
        self.model.eval()
        temps, vels = [], []
        temps_labels, vels_labels = [], []
        time_lim = min(len(dataset), max_time_limit)

        start = time.time()
        for timestep in range(0, time_lim, self.future_window):
            coords, temp, vel, dfun, temp_label, vel_label = dataset[timestep]
            coords = coords.to(self.device).float().unsqueeze(0)
            temp = temp.to(self.device).float().unsqueeze(0)
            vel = vel.to(self.device).float().unsqueeze(0)
            dfun = dfun.to(self.device).float().unsqueeze(0)
            
            # val doesn't apply push-forward
            temp_label = temp_label[0].to(self.device).float()
            vel_label = vel_label[0].to(self.device).float()
            with torch.no_grad():
                temp_pred, vel_pred = self._forward_int(coords[:, 0], temp[:, 0], vel[:, 0], dfun[:, 0])
                temp_pred = temp_pred.squeeze(0)
                vel_pred = vel_pred.squeeze(0)
                dataset.write_temp(temp_pred, timestep)
                dataset.write_vel(vel_pred, timestep)
                temps.append(temp_pred.detach().cpu())
                temps_labels.append(temp_label.detach().cpu())
                vels.append(vel_pred.detach().cpu())
                vels_labels.append(vel_label.detach().cpu())
        end = time.time()
        dur = end - start
        print(f'rollout time {dur:.4f} (s)')

        temps = torch.cat(temps, dim=0)
        temps_labels = torch.cat(temps_labels, dim=0)
        vels = torch.cat(vels, dim=0)
        vels_labels = torch.cat(vels_labels, dim=0)
        dfun = dataset.get_dfun()[:temps.size(0)]

        # print(temps.size(), temps_labels.size(), dfun.size())
        # print(vels.size(), vels_labels.size(), dfun.size())

        velx_preds = vels[0::2]
        velx_labels = vels_labels[0::2]
        vely_preds = vels[1::2]
        vely_labels = vels_labels[1::2]

        # print(temps.size(), temps_labels.size(), dfun.size())

        temp_metrics = compute_metrics(temps, temps_labels, dfun)
        print('TEMP METRICS')
        print(temp_metrics)
        velx_metrics = compute_metrics(velx_preds, velx_labels, dfun)
        print('VELX METRICS')
        print(velx_metrics)
        vely_metrics = compute_metrics(vely_preds, vely_labels, dfun)
        print('VELY METRICS')
        print(vely_metrics)

        # xgrid = dataset.get_x().permute((2, 0, 1))
        # print(heatflux(temps, dfun, self.val_variable, xgrid, dataset.get_dy()))
        # print(heatflux(labels, dfun, self.val_variable, xgrid, dataset.get_dy()))

        plt_iter_mae(temps, temps_labels, log_dir)
        plt_temp(temps, temps_labels, log_dir)

        def mag(velx, vely):
            return torch.sqrt(velx ** 2 + vely ** 2)

        mag_preds = mag(velx_preds, vely_preds)
        mag_labels = mag(velx_labels, vely_labels)

        plt_vel(mag_preds, mag_labels,
                velx_preds, velx_labels,
                vely_preds, vely_labels,
                log_dir)

        dataset.reset()
        return temp_metrics, velx_metrics, vely_metrics
