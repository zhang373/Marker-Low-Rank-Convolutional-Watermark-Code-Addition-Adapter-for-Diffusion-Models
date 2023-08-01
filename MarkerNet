from einops import rearrange, repeat
from ldm.modules.attention import CrossAttention,SpatialTransformer, FeedForward
from abc import abstractmethod
from functools import partial
import math
from typing import Iterable
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeEmb_Net(nn.Module):
    def __init__(self, Code, Adj_Code_hid_dim_1=30, Adj_Code_hid_dim_2=300):
        super().__init__()
        self.Code = [[]]
        self.Code_d_in = len(Code)
        self.src_code_len=3
        i=0 # 用后即弃
        for c in Code:
            self.Code[0].append([int(c), i, i/self.Code_d_in])
            i += 1
        self.Code=torch.tensor(self.Code, dtype=torch.float)
        #print(self.Code.shape)
        #self.featuremap = featuremap
        self.Code_hid_dim_1 = Adj_Code_hid_dim_1 * self.src_code_len
        self.Code_hid_dim_2 = Adj_Code_hid_dim_2 * self.src_code_len
        self.Code_d_out = self.Code_d_in
        self.Net = nn.Sequential(nn.Linear(3, self.Code_hid_dim_1),
                                 nn.ReLU(),
                                 nn.Linear(self.Code_hid_dim_1, self.Code_hid_dim_2),
                                 nn.Tanh(),
                                 nn.Linear(self.Code_hid_dim_2, self.Code_hid_dim_2)
                                 )

    def forward(self):
        CodeEmb = self.Net(self.Code)
        #CodeEmb = CodeEmb.unsqueeze(-1).unsqueeze(-1)
        #print("Size: ",CodeEmb.shape)
        #CodeEmb = CodeEmb.expand(-1, -1, self.featuremap.size(-2), self.featuremap.size(-1))
        #print(CodeEmb.shape)
        return CodeEmb


class CrossLowR_Net(nn.Module):
    def __init__(self, use_conv, Code, featuremap, Cr_channels_in, Adj_Cr_channels_Low, Adj_Cr_n_heads,
                 Adj_Cr_d_head, dims=2, padding='same'):
        super(CrossLowR_Net, self).__init__()
        self.Cr_channels_in = Cr_channels_in
        self.Cr_channels_out = self.Cr_channels_in
        self.Adj_Cr_channels_Low = Adj_Cr_channels_Low
        self.use_conv = use_conv
        self.dims = dims
        self.stride = 1
        Coder = CodeEmb_Net(Code)
        self.CodeEmb = Coder.forward()
        self.Code_dim = self.CodeEmb.shape[-1]
        print("self.Code_dim: ",self.Code_dim)
        self.Adj_Cr_n_heads = Adj_Cr_n_heads
        self.Adj_Cr_d_head = Adj_Cr_d_head

        if use_conv:
            self.op_down = conv_nd(
                dims, self.Cr_channels_in, self.Adj_Cr_channels_Low, 3, stride=self.stride, padding=padding
            )
            self.attn2 = SpatialTransformer(self.Adj_Cr_channels_Low, self.Adj_Cr_n_heads, self.Adj_Cr_d_head,
                                            depth=1, dropout=0., context_dim=self.Code_dim)  # is self-attn if context is none
            self.op_mid = conv_nd(
                dims, self.Adj_Cr_channels_Low, self.Adj_Cr_channels_Low, 3, stride=self.stride, padding=padding
            )
            self.op_up = conv_nd(
                dims, self.Adj_Cr_channels_Low, self.Cr_channels_in, 3, stride=self.stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=self.stride, stride=self.stride)

    def forward(self, x, Code):
        print("Check input", x.shape[1],self.Cr_channels_in)
        assert x.shape[1] == self.Cr_channels_in
        CodeNet = CodeEmb_Net(Code)
        CodeEmb = CodeNet.forward()
        #print("Code: ",CodeEmb)
        #print("CodeShape: ",CodeEmb.shape)
        #CodeEmb = CodeEmb.unsqueeze(-1).unsqueeze(-1)
        #print("CodeShape: ",CodeEmb.shape)
        #CodeEmb = CodeEmb.expand(-1, -1, self.featuremap.size(-2), self.featuremap.size(-1))
        x_in = x
        if self.use_conv:
            x = self.op_down(x)
            x = self.attn2(x, CodeEmb)
            x = self.op_mid(x)
            x = self.op_up(x)
            mask = x - x_in
            return x, mask
        else:
            return self.op(x)




shape = [4, 256//8, 256//8]
C, H, W = shape
batch_size = 1
size = (batch_size, C, H, W)
img = torch.randn(size)
code ="0010011"
TestCodeNet=CodeEmb_Net(code)
TestCodeNet.forward()
TestNet=CrossLowR_Net(use_conv=True, Code=code, featuremap=img, Cr_channels_in = 4, Adj_Cr_channels_Low = 2,
                      Adj_Cr_n_heads=4, Adj_Cr_d_head=4, dims=2, padding='same')
TestNet.forward(img,code)