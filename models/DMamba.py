import torch
from einops import rearrange, repeat
from torch import nn

from layers.embedding import FlexiPatchEmbed3d, UnPatchEmbed3d
from layers.mamba import Mamba3dLayer


class DMamba(nn.Module):
    def __init__(self, config, dataset_name='temp_input_dataset'):
        super().__init__()

        # Configuration
        self.time_window = config.time_window
        self.input_size = tuple(config.input_size)
        self.patch_size = tuple(config.patch_size)
        self.stride = tuple(config.stride)
        self.channels = config.channels
        self.dim = config.dim
        self.mlp_ratio = config.mlp_ratio

        # Do not use flexible patch size
        self.patch_size_seq = (tuple(config.patch_size),)
        self.patch_size_probs = None

        self.norm_layer = nn.LayerNorm
        self.depth = config.depth
        self.dropout = config.dropout
        self.dropout_embed = config.dropout_embed
        self.mamba_kw = config.get('mamba_kw', dict())
        self.training = False

        # Task
        self.dataset_name = dataset_name
        if self.dataset_name == 'temp_input_dataset':
            self.out_channels = 5
        elif self.dataset_name == 'vel_dataset':
            self.out_channels = 15

        # Patch Embedding
        self.patch_embedding1 = FlexiPatchEmbed3d(
            input_size=self.input_size,
            patch_size=self.patch_size,
            stride=self.stride,
            channels=1,
            d_embed=self.dim,
            patch_size_seq=self.patch_size_seq,
            patch_size_probs=self.patch_size_probs,
            norm_layer=self.norm_layer,
            bias=True,
        )
        self.patch_embedding2 = FlexiPatchEmbed3d(
            input_size=self.input_size,
            patch_size=self.patch_size,
            stride=self.stride,
            channels=2,  # channels * predict time window
            d_embed=self.dim,
            patch_size_seq=self.patch_size_seq,
            patch_size_probs=self.patch_size_probs,
            norm_layer=self.norm_layer,
            bias=True,
        )
        self.t = int((self.input_size[0] - self.patch_size[0]) / self.stride[0] + 1)
        self.h = int((self.input_size[1] - self.patch_size[1]) / self.stride[1] + 1)
        self.w = int((self.input_size[2] - self.patch_size[2]) / self.stride[2] + 1)

        # Backbone
        dims = [self.dim * (1 ** i) for i in range(self.depth + 1)]

        # Encoder layers
        self.encoder1 = nn.ModuleList(
            nn.Sequential(
                Mamba3dLayer(dim=dims[i], depth=2, attn_drop=self.dropout),
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.SiLU()
            ) for i in range(self.depth)
        )
        self.encoder2 = nn.ModuleList(
            nn.Sequential(
                Mamba3dLayer(dim=dims[i], depth=2, attn_drop=self.dropout),
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.SiLU()
            ) for i in range(self.depth)
        )

        # Bottleneck layer
        self.bridge = Mamba3dLayer(dim=dims[-1], depth=9, attn_drop=self.dropout)

        # Decoder layers
        self.decoder = nn.ModuleList(
            nn.Sequential(
                Mamba3dLayer(dim=dims[-(i + 1)], depth=2, attn_drop=self.dropout),
                nn.Linear(dims[-(i + 1)], dims[-(i + 2)]),
                nn.LayerNorm(dims[-(i + 2)]),
                nn.SiLU()
            ) for i in range(self.depth)
        )

        # Other layers
        self.dropout = nn.Dropout(self.dropout_embed)
        self.to_latent = nn.Identity()

        # Head
        if self.dataset_name == 'temp_input_dataset':
            self.head = UnPatchEmbed3d(
                patch_size=self.patch_size,
                stride=self.stride,
                # channels=self.out_channels,
                channels=1,
                d_embed=self.dim,
                bias=True
            )
        elif self.dataset_name == 'vel_dataset':
            self.head_temp = UnPatchEmbed3d(
                patch_size=self.patch_size,
                stride=self.stride,
                channels=5,
                d_embed=self.dim,
                bias=True
            )
            self.head_vel = UnPatchEmbed3d(
                patch_size=self.patch_size,
                stride=self.stride,
                channels=10,
                d_embed=self.dim,
                bias=True
            )

    def forward(self, x: torch.Tensor):
        # Predicting temperature
        if self.dataset_name == 'temp_input_dataset':
            temp = x[:, :self.time_window, ...]
            u = x[:, self.time_window:2 * self.time_window, ...]
            v = x[:, 3 * self.time_window:4 * self.time_window, ...]
            h = repeat(x[:, -2, ...], 'b h w -> b t h w', t=self.time_window)
            w = repeat(x[:, -1, ...], 'b h w -> b t h w', t=self.time_window)
            u_future = x[:, 2 * self.time_window:3 * self.time_window, ...]
            v_future = x[:, 4 * self.time_window:5 * self.time_window, ...]

        # Predicting temperature and velocity
        elif self.dataset_name == 'vel_dataset':
            temp = x[:, :self.time_window, ...]
            u = x[:, self.time_window:2 * self.time_window, ...]
            v = x[:, 2 * self.time_window:3 * self.time_window, ...]
            dfun = x[:, 3 * self.time_window:, ...]
            h = repeat(x[:, -2, ...], 'b h w -> b t h w', t=self.time_window)
            w = repeat(x[:, -1, ...], 'b h w -> b t h w', t=self.time_window)

        # Patch embeddings
        # temp = T(t)
        temp = temp.unsqueeze(dim=1)  # [B 1 T H W]
        temp = self.patch_embedding1(temp, return_patch_size=False, training=self.training)
        temp = self.dropout(temp)
        temp = rearrange(temp, 'b d t h w -> b t h w d')

        # uv1 = U(t)
        uv1 = torch.stack([u, v], dim=1)  # [B 2 T H W]
        uv1 = self.patch_embedding2(uv1, return_patch_size=False, training=self.training)
        uv1 = self.dropout(uv1)
        uv1 = rearrange(uv1, 'b d t h w -> b t h w d')

        # uv2 = U(t+1)
        uv2 = torch.stack([u_future, v_future], dim=1)  # [B 2 T H W]
        uv2 = self.patch_embedding2(uv2, return_patch_size=False, training=self.training)
        uv2 = self.dropout(uv2)
        uv2 = rearrange(uv2, 'b d t h w -> b t h w d')

        # Encoders
        temps = []
        for layer in self.encoder1:
            temp = layer(temp)
            temps.append(temp)

        uv1s = []
        for layer in self.encoder2:
            uv1 = layer(uv1)
            uv1s.append(uv1)

        uv2s = []
        for layer in self.encoder2:
            uv2 = layer(uv2)
            uv2s.append(uv2)

        # Bottleneck
        x = temp + uv1 + uv2
        x = self.bridge(x)

        # Decoder
        for i, layer in enumerate(self.decoder):
            x = x + temps[-(i + 1)]
            x = layer(x)

        x = self.to_latent(x)

        # Head
        x = rearrange(x, 'b t h w d -> b d t h w')
        if self.dataset_name == 'temp_input_dataset':
            x = self.head(x, patch_size=self.patch_size, stride=self.stride)
            return x.squeeze(dim=1)  # [B T H W]

        elif self.dataset_name == 'vel_dataset':
            x_temp = self.head_temp(x, patch_size=self.patch_size, stride=self.stride)
            x_vel = self.head_vel(x, patch_size=self.patch_size, stride=self.stride)
            x = torch.cat([x_temp, x_vel], dim=1)
            return x
