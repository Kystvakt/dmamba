import math
import random
import time

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from models.get_model import get_model
from op_lib import dist_util
from op_lib.disk_hdf5_dataset import DiskTempInputDataset, DiskTempVelDataset
from op_lib.hdf5_dataset import HDF5ConcatDataset, TempInputDataset, TempVelDataset
from op_lib.push_vel_trainer import PushVelTrainer
from op_lib.schedule_utils import LinearWarmupLR
from op_lib.temp_trainer import TempTrainer

# Select which dataset to use
torch_dataset_map = {
    'temp_input_dataset': (DiskTempInputDataset, TempInputDataset),
    'vel_dataset': (DiskTempVelDataset, TempVelDataset)
}

# Select which trainer to use
trainer_map = {
    'temp_input_dataset': TempTrainer,
    'vel_dataset': PushVelTrainer
}


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def build_datasets(cfg: DictConfig):
    dataset_class = torch_dataset_map[cfg.experiment.torch_dataset_name]
    time_window = cfg.experiment.train.time_window
    future_window = cfg.experiment.train.future_window
    push_forward_steps = cfg.experiment.train.push_forward_steps
    use_coords = cfg.experiment.train.use_coords
    # steady_time = cfg.dataset.steady_time

    # Normalize temperatures and velocities to [-1, 1]
    train_dataset = HDF5ConcatDataset([
        dataset_class[0](
            p,
            steady_time=cfg.dataset.steady_time,
            use_coords=use_coords,
            transform=cfg.dataset.transform,
            time_window=time_window,
            future_window=future_window,
            push_forward_steps=push_forward_steps
        ) for p in cfg.dataset.train_paths
    ])
    train_max_temp = train_dataset.normalize_temp_()
    train_max_vel = train_dataset.normalize_vel_()

    # Use same mapping as train dataset to normalize validation set
    val_dataset = HDF5ConcatDataset([
        dataset_class[1](
            p,
            steady_time=cfg.dataset.steady_time,
            use_coords=use_coords,
            time_window=time_window,
            future_window=future_window
        ) for p in cfg.dataset.val_paths
    ])
    val_dataset.normalize_temp_(train_max_temp)
    val_dataset.normalize_vel_(train_max_vel)

    assert val_dataset.absmax_temp() <= 1.5
    assert val_dataset.absmax_vel() <= 1.5

    return train_dataset, val_dataset, train_max_temp, train_max_vel


def build_dataloaders(train_dataset, val_dataset, cfg):
    if cfg.experiment.distributed:
        train_sampler = DistributedSampler(
            dataset=train_dataset,
            shuffle=cfg.experiment.train.shuffle_data
        )
        val_sampler = DistributedSampler(
            dataset=val_dataset,
            shuffle=False
        )
    else:
        train_sampler, val_sampler = None, None

    train_shuffle = cfg.experiment.train.shuffle_data and (train_sampler is None)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=train_shuffle,
        batch_size=cfg.experiment.train.batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=cfg.experiment.train.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2
    )
    return train_dataloader, val_dataloader


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(version_base=None, config_path='conf', config_name='default')
def train_app(cfg: DictConfig):
    # Configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg), end="\n\n")

    print("Training data files:")
    print(*cfg.dataset.train_paths, sep="\n", end="\n\n")

    assert cfg.test or cfg.train
    assert cfg.data_base_dir is not None
    assert cfg.log_dir is not None
    assert cfg.experiment.train.time_window > 0
    assert cfg.experiment.train.future_window > 0
    assert cfg.experiment.train.push_forward_steps > 0
    downsample_factor = cfg.experiment.train.downsample_factor
    if isinstance(downsample_factor, int):
        downsample_factor = [downsample_factor, downsample_factor]
    assert all([df >= 1 and isinstance(df, int) for df in downsample_factor])
    cfg.experiment.train.downsample_factor = downsample_factor

    exp = cfg.experiment
    model_name = exp.model.model_name.lower()

    if exp.distributed:
        dist_util.initialize('nccl')

    job_id = exp.exp_num
    if job_id:
        log_dir = f'{cfg.log_dir}/{cfg.dataset.name}/{exp.torch_dataset_name}/{model_name}_{job_id}'
    else:
        log_dir = f'{cfg.log_dir}'

    writer = SummaryWriter(log_dir=log_dir)

    # Fix seed
    fix_seed(exp.seed)

    # Dataset
    train_dataset, val_dataset, train_max_temp, train_max_vel = build_datasets(cfg)
    train_dataloader, val_dataloader = build_dataloaders(train_dataset, val_dataset, cfg)
    print("Train data size:", len(train_dataloader))
    val_variable = 0

    in_channels = train_dataset.datasets[0].in_channels
    out_channels = train_dataset.datasets[0].out_channels

    # Domain_rows and domain_cols are used to determine the number of modes used in fourier models.
    _, domain_rows, domain_cols = train_dataset.datum_dim()
    downsampled_rows = domain_rows / downsample_factor[0]
    downsampled_cols = domain_cols / downsample_factor[1]

    # Model
    model = get_model(
        model_name,
        in_channels,
        out_channels,
        downsampled_rows,
        downsampled_cols,
        exp,
        exp.model.device
    )
    if cfg.model_checkpoint:
        model.load_state_dict(torch.load(cfg.model_checkpoint))
    n_params = count_params(model)
    print(f"Model has {n_params} trainable parameters.")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=dist_util.world_size() * exp.optimizer.initial_lr,
        weight_decay=exp.optimizer.weight_decay
    )

    # Scheduler
    total_iters = exp.train.max_epochs * len(train_dataloader)
    warmup_iters = max(1, int(math.sqrt(dist_util.world_size()) * 0.03 * total_iters))
    warmup_lr = LinearWarmupLR(optimizer, warmup_iters)
    warm_iters = total_iters - warmup_iters
    if exp.lr_scheduler.name == 'step':
        # Step size is scaled by the size of dataloader,
        # because we check each step so that it's compatible with cosine scheduler.
        warm_schedule = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=exp.lr_scheduler.patience * len(train_dataloader),
            gamma=exp.lr_scheduler.factor
        )
    elif exp.lr_scheduler.name == 'cosine':
        warm_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=exp.train.max_epochs * len(train_dataloader),
            eta_min=exp.lr_scheduler.eta_min
        )
    else:
        raise NotImplementedError(f"Unsupported scheduler type: {exp.lr_scheduler.name}")
    # SequentialLR produces a deprecation warning when calling sub-schedulers.
    # https://github.com/pytorch/pytorch/issues/76113
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_lr, warm_schedule], [warmup_iters])

    # Trainer
    trainer_class = trainer_map[exp.torch_dataset_name]
    trainer = trainer_class(
        model,
        exp.train.future_window,
        exp.train.push_forward_steps,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        val_variable,
        writer,
        exp
    )
    # print(trainer)

    # Logging
    if exp.log_to_wandb:
        wandb.login(key=exp.wandb.api_key)
        project_name = f"{cfg.dataset.name}_{exp.torch_dataset_name.split('_')[0]}"
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=project_name,
            name=f"{model_name}_{exp.exp_num}",
            config=config_dict
        )

    # Training
    if cfg.train and not cfg.model_checkpoint:
        ckpt_path = trainer.train(exp.train.max_epochs, log_dir, dataset_name=cfg.dataset.name)
    elif not cfg.train and cfg.model_checkpoint:
        ckpt_path = cfg.model_checkpoint
    else:
        raise ValueError(
            "Invalid configuration: Neither training mode is active nor a valid model checkpoint is provided. "
            "Please either set cfg.train to True for training or provide a valid model checkpoint path in "
            "cfg.model_checkpoint for testing."
        )

    # Test with the best model
    if cfg.test and dist_util.is_leader_process():
        model_state_dict = torch.load(ckpt_path)
        trainer.model.load_state_dict(model_state_dict)
        metrics = trainer.test(val_dataset.datasets[0], log_dir)

        save_dict = {
            'id': f'{cfg.dataset.name}_{model_name}_{exp.torch_dataset_name}',
            'metrics': metrics,
            # used for normalization
            'train_data_max_temp': train_max_temp,
            'train_data_max_vel': train_max_vel,
            # can be used to restart the learning rate
            'epochs': exp.train.max_epochs,
            # info needed to reconstruct the model
            'model_state_dict': model.state_dict(),
            'in_channels': in_channels,
            'out_channels': out_channels,
            'downsampled_rows': downsampled_rows,
            'downsampled_cols': downsampled_cols,
            'exp': exp,
        }
        torch.save(save_dict, f'{ckpt_path}')

    if exp.log_to_wandb:
        wandb.finish()


if __name__ == '__main__':
    train_app()
