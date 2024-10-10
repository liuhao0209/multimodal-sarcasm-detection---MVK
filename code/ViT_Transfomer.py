"""
Adapted from https://github.com/lukemelas/simple-bert
"""

import numpy as np
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import torch


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None  # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class PositionWiseFeedForward_change_dim(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, dim, ff_dim,outdim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, outdim)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(torch.relu(self.fc1(x)))

class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class ViTTransformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(self, num_layers, dims, num_heads, ff_dim, dropout):
        super(ViTTransformer,self).__init__()
        self.dim = 512
        self.blocks = nn.ModuleList([
            Block(self.dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.dim_convert0 = nn.Linear(dims,self.dim)
        self.dim_convert1 = nn.Linear(self.dim,dims)
    def forward(self, x, mask=None):
        x = torch.relu(self.dim_convert0(x))
        for block in self.blocks:
            x = block(x, mask)
        x = torch.relu(self.dim_convert1(x))
        return x


#--------------------------------------------------------------

class crossmodal_block(nn.Module):
    def __init__(self, dim, dim2,dim_out, num_heads, ff_dim, dropout):
        super(crossmodal_block, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=0.2)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward_change_dim(dim, ff_dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.y_linear = nn.Linear(dim2,dim)
        self.modal_linear = nn.Linear(dim, dim_out)

    def forward(self, x, y):

        y = self.y_linear(y)

        h = self.drop(self.norm1(self.proj( self.attn(x, y, y)[0] )))
        x = x + h
        #hidden = x

        h = self.drop(self.norm2(self.pwff(x)))
        hidden = h
        x = x + h
        x = self.drop(self.modal_linear(self.norm3(x)))
        return x, hidden


class TowertransformerEncoder(nn.Module):
    def __init__(self,num_layers,dims, dims2, dim_out, num_heads,ff_dim,dropout=0.2):
        super(TowertransformerEncoder, self).__init__()
        self.dim = dims
        # self.blocks = nn.ModuleList([
        #     crossmodal_block(self.dim, num_heads=4, ff_dim=512, dropout=0.2) for _ in range(num_layers)
        # ])
        self.block = crossmodal_block(self.dim, dim2 =dims2, dim_out=dim_out, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)

    def forward(self,modality_x,modality_y):
        #modality_x = torch.unsqueeze(modality_x, dim=0)
        #modality_y = torch.unsqueeze(modality_y, dim=0)
        modality_x, hidden = self.block(modality_x,modality_y) # x主

        modality_x = torch.squeeze(modality_x, dim=0)
        #modality_y = torch.squeeze (modality_y, dim=0)
        return modality_x, hidden




if __name__ == '__main__':
    # inputs = torch.randn(196, 32, 500)
    # trans = ViTTransformer(num_layers=6, dims=500, num_heads=8, ff_dim=512, dropout=0.2)
    # out = trans(inputs, mask=None)
    # print("out ", out.size())

    inputi = torch.randn( 32, 505)
    inputt = torch.randn( 32, 712)
    inputa = torch.randn( 32, 205)
    transi = TowertransformerEncoder(num_layers=6, dims=505, dims2=712,dim_out=500, num_heads=5, ff_dim=512, dropout=0.2)
    transa = TowertransformerEncoder(num_layers=6, dims=205, dims2=712,dim_out=200, num_heads=5, ff_dim=512, dropout=0.2)
    transt = TowertransformerEncoder(num_layers=6, dims=712, dims2=710, dim_out=512, num_heads=4, ff_dim=512, dropout=0.2)
    outi1, outi2 = transi(inputi, inputt)
    outa1, outa2 = transa(inputa, inputt)
    outt1, outt2 = transt(inputt, torch.cat((inputi,inputa ), dim=-1))
    print(" outi ",outi1.size(), outi2.size())
    print(" outa ", outa1.size(), outa2.size())
    print(" outt ", outt1.size(), outt2.size())


