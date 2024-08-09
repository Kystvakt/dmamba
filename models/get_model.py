import os
from torch.nn.parallel import DistributedDataParallel as DDP
from models.unet_arena import Unet
from models.unet_bench import UNet2d
from neuralop.models import FNO, UNO
from .gefno.gfno import GFNO2d
from models.DMamba import DMamba
from models.Transolver import PhysicsAttentionStructuredMesh3D as Transolver

from .factorized_fno.factorized_fno import FNOFactorized2DBlock 



# Model names
_UNET_BENCH = 'unet_bench'
_UNET_ARENA = 'unet_arena'
_UFNET = 'ufnet'
_FNO = 'fno'
_UNO = 'uno'
_FFNO = 'factorized_fno'
_GFNO = 'gfno'
_CNO = 'cno'
_DMamba = 'dmamba'
_Transolver = 'transolver'

# Model list
_MODEL_LIST = [
    _UNET_BENCH,
    _UNET_ARENA,
    _UFNET,
    _FNO,
    _UNO,
    _FFNO,
    _GFNO,
    _CNO,
    _DMamba,
    _Transolver
]


def get_model(
        model_name,
        in_channels,
        out_channels,
        domain_rows,
        domain_cols,
        exp,
        device
):
    assert model_name in _MODEL_LIST, f'Model name {model_name} invalid'

    if model_name == _UNET_ARENA:
        model = Unet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=exp.model.hidden_channels,
            ch_mults=[1, 2, 2, 4, 4],
            is_attn=[False] * 5,
            activation='gelu',
            mid_attn=False,
            norm=True,
            use1x1=True
        )
    elif model_name == _UNET_BENCH:
        model = UNet2d(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=exp.model.init_features
        )
    elif model_name == _FNO:
        model = FNO(
            n_modes=(exp.model.modes, exp.model.modes),
            hidden_channels=exp.model.hidden_channels,
            domain_padding=exp.model.domain_padding[0],
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=exp.model.n_layers,
            norm=exp.model.norm,
            rank=exp.model.rank,
            factorization='tucker',
            implementation='factorized',
            separable=False
        )
    elif model_name == _GFNO:
        model = GFNO2d(in_channels=in_channels,
                       out_channels=out_channels,
                       modes=exp.model.modes // 2,
                       width=exp.model.width,
                       reflection=exp.model.reflection,
                       domain_padding=exp.model.domain_padding) # padding is NEW
    
    elif model_name == _FFNO:
        model = FNOFactorized2DBlock(in_channels=in_channels,
                                     out_channels=out_channels,
                                     modes=exp.model.modes // 2,
                                     width=exp.model.width,
                                     dropout=exp.model.dropout,
                                     n_layers=exp.model.n_layers)
    elif model_name == _UNO:
        model = UNO(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=exp.model.hidden_channels,
            projection_channels=exp.model.projection_channels,
            uno_out_channels=exp.model.uno_out_channels,
            uno_n_modes=exp.model.uno_n_modes,
            uno_scalings=exp.model.uno_scalings,
            n_layers=exp.model.n_layers,
            domain_padding=exp.model.domain_padding
        )
    elif model_name == _DMamba:
        model = DMamba(exp.model, exp.train.time_window, exp.torch_dataset_name)
    elif model_name == _Transolver:
        model = Transolver(dataset_name=exp.torch_dataset_name)
    else:
        raise NotImplementedError

    if exp.distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        model = model.to(local_rank).float()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        model = model.to(device).float()

    return model
