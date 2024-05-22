from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
import wandb
import time

from metrics import compute_metrics, write_metrics
from losses import LpLoss
from plt_util import plt_temp, plt_iter_mae
from dist_util import local_rank, is_leader_process
from downsample import downsample_domain


class TempTrainer:
    def __init__(
            self,
            model,
            future_window,
            push_forward_steps,
            train_dataloader,
            val_dataloader,
            optimizer,
            lr_scheduler,
            val_variable,
            writer,
            cfg
    ):
        # configuration
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.val_variable = val_variable
        self.writer = writer
        self.cfg = cfg
        self.device = self.cfg.model.device

        # logging
        self.log_to_wandb = self.cfg.log_to_wandb
        if self.log_to_wandb:
            self.wandb_config = self.cfg.wandb

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
        ckpt_root = f'{log_dir}/{dataset_name}_{self.cfg.torch_dataset_name.split("_")[0]}'
        Path(ckpt_root).mkdir(parents=True, exist_ok=True)
        ckpt_path = f'{ckpt_root}/{ckpt_file}'
        print(f'saving model to {ckpt_path}')

        if self.cfg.distributed:
            torch.save(self.model.module.state_dict(), f'{ckpt_path}')
        else:
            torch.save(self.model.state_dict(), f'{ckpt_path}')
        return ckpt_path

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

            val_dataset = self.val_dataloader.dataset.datasets[0]

            metrics = self.test(val_dataset)
            if metrics.rmse < best_rmse:
                best_rmse = metrics.rmse
                save_metrics = metrics
                ckpt_path = self.save_checkpoint(log_dir, dataset_name, epoch)

            if self.log_to_wandb:
                test_logs = defaultdict(float)
                test_logs['MAE'] = metrics.mae
                test_logs['RMSE'] = metrics.rmse
                test_logs['Relative Error'] = metrics.relative_error
                test_logs['Max Error'] = metrics.max_error
                test_logs['Boundary RMSE'] = metrics.boundary_rmse
                test_logs['Interface RMSE'] = metrics.interface_rmse
                test_logs['Fourier - Low'] = metrics.fourier_low
                test_logs['Fourier - Mid'] = metrics.fourier_mid
                test_logs['Fourier - High'] = metrics.fourier_high
                wandb.log(test_logs, step=epoch)

        if self.log_to_wandb:
            wandb.finish()
        return save_metrics, ckpt_path

    def _forward_int(self, coords, temp, vel):
        input_ = torch.cat((temp, vel), dim=1)
        if self.cfg.train.use_coords:
            input_ = torch.cat((coords, input_), dim=1)
        pred = self.model(input_)
        return pred

    def push_forward_trick(self, coords, temp, vel):
        if self.cfg.train.noise:
            temp += torch.empty_like(temp).normal_(0, 0.01)
            vel += torch.empty_like(vel).normal_(0, 0.01)
        pred = self._forward_int(coords, temp, vel)
        return pred

    def train_step(self, epoch):
        self.model.train()

        epoch_loss = 0
        epoch_mse_loss = 0

        for i, (coords, temp, vel, label) in enumerate(self.train_dataloader):
            coords = coords.to(self.device).float()
            temp = temp.to(self.device).float()
            vel = vel.to(self.device).float()
            label = label.to(self.device).float()
            coords, temp, vel, label = downsample_domain(self.cfg.train.downsample_factor, coords, temp, vel, label)  #

            pred = self.push_forward_trick(coords, temp, vel)

            loss = self.loss(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            mse_loss = F.mse_loss(pred, label).detach()

            epoch_loss += loss
            epoch_mse_loss += mse_loss
            global_iter = epoch * len(self.train_dataloader) + i
            write_metrics(pred, label, global_iter, 'Train', self.writer)
            del temp, vel, label

        print((
            f'Epoch : {epoch} | '
            f'train loss: {epoch_loss / len(self.train_dataloader)}, '
            f'mse: {epoch_mse_loss / len(self.train_dataloader)}'
        ))
        if self.log_to_wandb:
            train_logs = defaultdict(float)
            train_logs['train_loss'] = epoch_loss / len(self.train_dataloader)
            train_logs['train_mse_loss'] = epoch_mse_loss / len(self.train_dataloader)
            wandb.log(train_logs, step=epoch)

    def val_step(self, epoch):
        val_epoch_loss = 0
        self.model.eval()
        for i, (coords, temp, vel, label) in enumerate(self.val_dataloader):

            coords = coords.to(self.device).float()
            temp = temp.to(self.device).float()
            vel = vel.to(self.device).float()
            label = label.to(self.device).float()

            with torch.no_grad():
                pred = self._forward_int(coords, temp, vel)
                temp_loss = F.mse_loss(pred, label)
                loss = temp_loss

            val_epoch_loss += loss

            global_iter = epoch * len(self.val_dataloader) + i
            write_metrics(pred, label, global_iter, 'Val', self.writer)
            del temp, vel, label

        print(f'Epoch : {epoch} | val loss: {val_epoch_loss / len(self.val_dataloader)}')
        if self.log_to_wandb:
            val_logs = defaultdict(float)
            val_logs['val_loss'] = val_epoch_loss / len(self.val_dataloader)
            wandb.log(val_logs, step=epoch)

    def test(self, dataset, max_timestep=200):
        if is_leader_process():
            self.model.eval()
            temps = []
            labels = []
            time_lim = min(len(dataset), max_timestep)

            start = time.time()
            for timestep in range(0, time_lim, self.future_window):
                coords, temp, vel, label = dataset[timestep]

                coords = coords.to(self.device).float().unsqueeze(0)
                temp = temp.to(self.device).float().unsqueeze(0)
                vel = vel.to(self.device).float().unsqueeze(0)
                label = label.to(self.device).float()

                with torch.no_grad():
                    pred = self._forward_int(coords, temp, vel)
                    temp = F.hardtanh(pred.squeeze(0), -1, 1)
                    dataset.write_temp(temp, timestep)
                    temps.append(temp.detach().cpu())
                    labels.append(label.detach().cpu())
            dur = time.time() - start
            print(f'rollout time {dur} (s)')

            temps = torch.cat(temps, dim=0)
            labels = torch.cat(labels, dim=0)
            dfun = dataset.get_dfun()[:temps.size(0)]

            metrics = compute_metrics(temps, labels, dfun)
            print(metrics)

            plt_temp(temps, labels, self.model.__class__.__name__)
            plt_iter_mae(temps, labels)

            dataset.reset()

            return metrics
