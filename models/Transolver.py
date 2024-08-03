import torch
import torch.nn as nn
from einops import rearrange, repeat


class PhysicsAttentionStructuredMesh3D(nn.Module):
    """For structured mesh in 3D space."""
    def __init__(self, dataset_name='temp_input_dataset', in_dim=None, out_dim=None, heads=8, dim_head=64, dropout=0., slice_num=32, H=32, W=32, D=32, kernel=3):
        super().__init__()
        self.dataset_name = dataset_name
        if dataset_name == 'temp_input_dataset':
            in_dim = 7
            out_dim = 1
        elif dataset_name == 'vel_dataset':
            in_dim = 5
            out_dim = 2
        self.time_window = 5
        # self.future_window = 5

        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.D = D

        self.in_project_x = nn.Conv3d(in_dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(in_dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Predicting temperature
        if self.dataset_name == 'temp_input_dataset':
            temp = x[:, :self.time_window, ...]
            u = x[:, self.time_window:2 * self.time_window, ...]
            v = x[:, 3 * self.time_window:4 * self.time_window, ...]
            h = repeat(x[:, -2, ...], 'b h w -> b t h w', t=self.time_window)
            w = repeat(x[:, -1, ...], 'b h w -> b t h w', t=self.time_window)
            u_future = x[:, 2 * self.time_window:3 * self.time_window, ...]
            v_future = x[:, 4 * self.time_window:5 * self.time_window, ...]

            x = torch.stack([temp, u, v, h, w, u_future, v_future], dim=1)  # (B, C, T, H, W)
            B, C, T, H, W = x.shape
            N = T * H * W

        # Predicting temperature and velocity
        elif self.dataset_name == 'vel_dataset':
            temp = x[:, :self.time_window, ...]
            u = x[:, self.time_window:2 * self.time_window, ...]
            v = x[:, 2 * self.time_window:3 * self.time_window, ...]
            dfun = x[:, 3 * self.time_window:, ...]
            h = repeat(x[:, -2, ...], 'b h w -> b t h w', t=self.time_window)
            w = repeat(x[:, -1, ...], 'b h w -> b t h w', t=self.time_window)

            x = torch.stack([temp, u, v, h, w], dim=1)  # (B, C, T, H, W)
            B, C, T, H, W = x.shape
            N = T * H * W

        else:
            raise ValueError(f"Dataset unsupported: {self.dataset_name}")

        # B N C
        # B, N, C = x.shape
        # x = x.reshape(B, self.H, self.W, self.D, C).contiguous().permute(0, 4, 1, 2, 3).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        out_x = self.to_out(out_x)

        out_x = rearrange(out_x, 'b (h w t) d -> b d t h w', h=H, w=W, t=T)

        if self.dataset_name == 'temp_input_dataset':
            out_x = out_x.squeeze(dim=1)

        return out_x
