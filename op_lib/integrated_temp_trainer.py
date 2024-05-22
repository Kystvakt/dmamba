from collections import defaultdict
from pathlib import Path
import os
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, PolynomialLR
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import wandb

from .hdf5_dataset import HDF5Dataset, TempVelDataset
from .metrics import compute_metrics, write_metrics
from .losses import LpLoss
from .plt_util import plt_temp, plt_iter_mae
from .heatflux import heatflux
from .dist_util import local_rank, is_leader_process
from .downsample import downsample_domain

from torch.cuda import nvtx
import time

t_bulk_map = {
    'wall_super_heat': 58,
    'subcooled': 50
}


class TempTrainer:
    def __init__(self,
                 model,
                 future_window,
                 push_forward_steps,
                 train_dataloader_list,
                 val_dataloader_list,
                 optimizer,
                 lr_scheduler,
                 val_variable,
                 writer,
                 cfg):
        self.model = model
        self.train_dataloader_list = train_dataloader_list
        self.val_dataloader_list = val_dataloader_list
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.val_variable = val_variable
        self.writer = writer
        self.cfg = cfg
        # MH (Start)
        self.device = self.cfg.model.device
        self.log_to_wandb = self.cfg.log_to_wandb
        # logging
        if self.log_to_wandb:
            self.wandb_config = self.cfg.wandb
        # MH (End)
        self.loss = LpLoss(d=2, reduce_dims=[0, 1])

        self.push_forward_steps = push_forward_steps
        self.future_window = future_window
        self.local_rank = local_rank()

    def save_checkpoint(self, log_dir, dataset_name, epoch):
        timestamp = int(time.time())
        if self.cfg.distributed:
            model_name = self.model.module.__class__.__name__
        else:
            model_name = self.model.__class__.__name__
        ckpt_file = f'{model_name}_{self.cfg.torch_dataset_name}_{self.cfg.model.model_num}.pt'
        # ckpt_file = f'{model_name}_{self.cfg.torch_dataset_name}_{self.cfg.train.max_epochs}_{timestamp}.pt'
        # ckpt_root = Path.home() / f'{log_dir}/{dataset_name}'
        ckpt_root = f'{log_dir}/{dataset_name}_{self.cfg.torch_dataset_name.split("_")[0]}'
        Path(ckpt_root).mkdir(parents=True, exist_ok=True)
        ckpt_path = f'{ckpt_root}/{ckpt_file}'
        print(f'saving model to {ckpt_path}')
        if self.cfg.distributed:
            torch.save(self.model.module.state_dict(), f'{ckpt_path}')
        else:
            torch.save(self.model.state_dict(), f'{ckpt_path}')
        return ckpt_path

    # def train(self, max_epochs, *args, **kwargs):
    def train(self, max_epochs, log_dir, dataset_name):

        if self.log_to_wandb:
            wandb.init(project=self.wandb_config.project, name=self.cfg.model.model_name.lower())

        best_rmse = 1e+5
        save_metrics = {}
        ckpt_path = ''

        for epoch in range(max_epochs):
            print('epoch ', epoch)
            self.train_step(epoch)
            self.val_step(epoch)
            # test each epoch
            total_rmse = 0
            metrics_list = []
            for idx, val_dataloader in enumerate(self.val_dataloader_list):
                val_dataset = val_dataloader.dataset.datasets[0]
                # self.test(val_dataset)
                # MH (Start)
                metrics = self.test(val_dataset, idx)
                total_rmse += metrics.rmse
                metrics_list.append(metrics)

            if total_rmse < best_rmse:
                best_rmse = total_rmse
                save_metrics = metrics_list
                ckpt_path = self.save_checkpoint(log_dir, dataset_name, epoch)

            if self.log_to_wandb:
                test_logs = defaultdict(float)
                test_logs['MAE'] = sum([metrics.mae for metrics in metrics_list]) / len(metrics_list)
                test_logs['RMSE'] = sum([metrics.rmse for metrics in metrics_list]) / len(metrics_list)
                test_logs['Relative Error'] = sum([metrics.relative_error for metrics in metrics_list]) / len(
                    metrics_list)
                test_logs['Max Error'] = sum([metrics.max_error for metrics in metrics_list]) / len(metrics_list)
                test_logs['Boundary RMSE'] = sum([metrics.boundary_rmse for metrics in metrics_list]) / len(
                    metrics_list)
                test_logs['Interface RMSE'] = sum([metrics.interface_rmse for metrics in metrics_list]) / len(
                    metrics_list)
                test_logs['Fourier - Low'] = sum([metrics.fourier_los for metrics in metrics_list]) / len(metrics_list)
                test_logs['Fourier - Mid'] = sum([metrics.fourier_mid for metrics in metrics_list]) / len(metrics_list)
                test_logs['Fourier - High'] = sum([metrics.fourier_high for metrics in metrics_list]) / len(
                    metrics_list)
                wandb.log(test_logs, step=epoch)
            # MH (End)

        if self.log_to_wandb:
            wandb.finish()
        return save_metrics, ckpt_path

    def _forward_int(self, coords, temp, vel, idx):
        input = torch.cat((temp, vel), dim=1)
        if self.cfg.train.use_coords:
            input = torch.cat((coords, input), dim=1)
        pred = self.model(input, idx)
        return pred

    def push_forward_trick(self, coords, temp, vel, idx):
        if self.cfg.train.noise:
            temp += torch.empty_like(temp).normal_(0, 0.01)
            vel += torch.empty_like(vel).normal_(0, 0.01)
        pred = self._forward_int(coords, temp, vel, idx)
        return pred

    # sciml/op_lib/temp_trainer.py line 78
    def train_step(self, epoch):
        self.model.train()
        # MH (Start)
        epoch_loss = 0
        epoch_mse_loss = 0
        # MH (End)
        for idx, train_dataloader in enumerate(self.train_dataloader_list):
            batch_loss = 0
            batch_mse_loss = 0
            for iter, (coords, temp, vel, label) in enumerate(train_dataloader):
                # coords = coords.to(self.local_rank).float()
                # temp = temp.to(self.local_rank).float()
                # vel = vel.to(self.local_rank).float()
                # label = label.to(self.local_rank).float()
                # MH (Start)
                coords = coords.to(self.device).float()
                temp = temp.to(self.device).float()
                vel = vel.to(self.device).float()
                label = label.to(self.device).float()
                # MH (End)
                coords, temp, vel, label = downsample_domain(self.cfg.train.downsample_factor, coords, temp, vel,
                                                             label)  #

                pred = self.push_forward_trick(coords, temp, vel, idx)

                # print(pred.size(), label.size())

                loss = self.loss(pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()

                mse_loss = F.mse_loss(pred, label).detach()
                # MH (Start)
                batch_loss += loss
                batch_mse_loss += mse_loss
                # MH (End)
                # print(f'train loss: {loss}, mse: {mse_loss}')
                global_iter = epoch * len(train_dataloader) + iter
                write_metrics(pred, label, global_iter, 'Train', self.writer)
                del temp, vel, label

            epoch_loss += batch_loss
            epoch_mse_loss += batch_mse_loss
            print(
                f'Dataset : {idx + 1} | Epoch : {epoch} | train loss: {epoch_loss / len(train_dataloader)}, mse: {epoch_mse_loss / len(train_dataloader)}')
        print(
            f'Epoch : {epoch} | train loss: {epoch_loss / sum([len(train_dataloader) for train_dataloader in self.train_dataloader_list])}, mse: {epoch_mse_loss / sum([len(train_dataloader) for train_dataloader in self.train_dataloader_list])}')
        if self.log_to_wandb:
            train_logs = defaultdict(float)
            train_logs['train_loss'] = epoch_loss / sum(
                [len(train_dataloader) for train_dataloader in self.train_dataloader_list])
            train_logs['train_mse_loss'] = epoch_mse_loss / sum(
                [len(train_dataloader) for train_dataloader in self.train_dataloader_list])
            wandb.log(train_logs, step=epoch)

    def val_step(self, epoch):
        # MH (Start)
        val_epoch_loss = 0
        # MH (End)
        self.model.eval()
        for idx, val_dataloader in enumerate(self.val_dataloader_list):
            val_batch_loss = 0
            for iter, (coords, temp, vel, label) in enumerate(val_dataloader):
                # MH (Start)
                coords = coords.to(self.device).float()
                temp = temp.to(self.device).float()
                vel = vel.to(self.device).float()
                label = label.to(self.device).float()
                # MH (End)
                with torch.no_grad():
                    pred = self._forward_int(coords, temp, vel, idx)
                    temp_loss = F.mse_loss(pred, label)
                    loss = temp_loss
                # MH (Start)
                val_batch_loss += loss
                # MH (End)
                # print(f'val loss: {loss}')
                global_iter = epoch * len(val_dataloader) + iter
                write_metrics(pred, label, global_iter, 'Val', self.writer)
                del temp, vel, label

            val_epoch_loss += val_batch_loss
            print(f'Dataset : {idx + 1} | Epoch : {epoch} | val loss: {val_epoch_loss / len(val_dataloader)}')
            if self.log_to_wandb:
                val_logs = defaultdict(float)
                val_logs['val_loss'] = val_epoch_loss / len(self.val_dataloader)
                wandb.log(val_logs, step=epoch)
        print(
            f'Epoch : {epoch} | val loss: {val_epoch_loss / sum([len(val_dataloader) for val_dataloader in self.val_dataloader_list])}')

    def test(self, dataset, idx, max_timestep=200):
        if is_leader_process():
            self.model.eval()
            temps = []
            labels = []
            time_lim = min(len(dataset), max_timestep)

            start = time.time()
            for timestep in range(0, time_lim, self.future_window):
                coords, temp, vel, label = dataset[timestep]
                # coords = coords.to(self.local_rank).float().unsqueeze(0)
                # temp = temp.to(self.local_rank).float().unsqueeze(0)
                # vel = vel.to(self.local_rank).float().unsqueeze(0)
                # label = label.to(self.local_rank).float()
                # MH (Start)
                coords = coords.to(self.device).float().unsqueeze(0)
                temp = temp.to(self.device).float().unsqueeze(0)
                vel = vel.to(self.device).float().unsqueeze(0)
                label = label.to(self.device).float()
                # MH (End)
                with torch.no_grad():
                    pred = self._forward_int(coords, temp, vel, idx)
                    temp = F.hardtanh(pred.squeeze(0), -1, 1)
                    dataset.write_temp(temp, timestep)
                    temps.append(temp.detach().cpu())
                    labels.append(label.detach().cpu())
            dur = time.time() - start
            print(f'rollout time {dur} (s)')

            temps = torch.cat(temps, dim=0)
            labels = torch.cat(labels, dim=0)
            dfun = dataset.get_dfun()[:temps.size(0)]

            # print(temps.max(), temps.min())
            # print(labels.max(), labels.min())

            metrics = compute_metrics(temps, labels, dfun)
            print(metrics)

            # xgrid = dataset.get_x().permute((2, 0, 1))
            # print(heatflux(temps, dfun, self.val_variable, xgrid, dataset.get_dy()))
            # print(heatflux(labels, dfun, self.val_variable, xgrid, dataset.get_dy()))

            plt_temp(temps, labels, self.model.__class__.__name__)
            plt_iter_mae(temps, labels)

            dataset.reset()

            return metrics
