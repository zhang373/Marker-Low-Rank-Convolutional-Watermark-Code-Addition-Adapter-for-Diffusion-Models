from einops import rearrange, repeat
from ldm.modules.attention import CrossAttention,SpatialTransformer, FeedForward
from ldm.modules.diffusionmodules.openaimodel import Downsample, Upsample
from abc import abstractmethod
from functools import partial
import math
from typing import Iterable
import pytorch_ssim
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
    def __init__(self, Code, batchsize, Adj_Code_hid_dim_1=30, Adj_Code_hid_dim_2=300):
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
        self.batch_size = batchsize
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
        return CodeEmb.repeat(self.batch_size, 1, 1)

class CrossLowR_Net(nn.Module):
    def __init__(self, use_conv,batchsize, Code, Cr_channels_in, Adj_Cr_channels_Low, Adj_Cr_n_heads=8,
                 Adj_Cr_d_head=8, dims=2, padding='same'):
        super(CrossLowR_Net, self).__init__()
        self.Cr_channels_in = Cr_channels_in
        self.Cr_channels_out = self.Cr_channels_in
        self.Adj_Cr_channels_Low = Adj_Cr_channels_Low
        self.use_conv = use_conv
        self.dims = dims
        self.stride = 1
        Coder = CodeEmb_Net(Code,batchsize)
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

    def forward(self, x, Code, bs):
        print("Check input", x.shape[1],self.Cr_channels_in)
        assert x.shape[1] == self.Cr_channels_in
        CodeNet = CodeEmb_Net(Code,batchsize= bs)
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

class Marker_pretrain_Net(nn.Module):
    def __init__(self, Channel_in, Channel_mid, Code, Adj_Cr_channels_Low):
        super(Marker_pretrain_Net, self).__init__()
        self.Code = Code
        self.Adj_Cr_channels_Low = Adj_Cr_channels_Low
        self.Channel_in = Channel_in
        self.Channel_mid = Channel_mid
        #self.Img_H = Img_H
        #self.Img_W = Img_W
        # for scale factor, please refer to "https://pic3.zhimg.com/v2-1a60fadfd1b8cb1b41bad5f7deddf526_r.jpg"
        # Also: https://pic1.zhimg.com/v2-7476bc68a913afd13a2e4483a8869a04_r.jpg
        self.ChannelScaleFactor = [1, 2, 4]
        self.SizeScaleFactor = [1, 1/2, 1/4, 1/8]

        # For Down Blocks
        self.Down_Net_1 = conv_nd(2, self.Channel_in, self.Channel_mid, 3, padding=1)
        self.Down_Net_2 = CrossLowR_Net(True, self.Code, self.Channel_mid, self.Adj_Cr_channels_Low)
        self.DownSample_Net_1 = Downsample( self.Channel_mid**self.ChannelScaleFactor[0], True,
                                            dims=2, out_channels=self.Channel_mid*self.ChannelScaleFactor[1])

        self.Down_Net_3 = CrossLowR_Net(True, self.Code, self.Channel_mid*self.ChannelScaleFactor[1],
                                        self.Adj_Cr_channels_Low)
        self.DownSample_Net_2 = Downsample(self.Channel_mid ** self.ChannelScaleFactor[1], True,
                                           dims=2, out_channels=self.Channel_mid * self.ChannelScaleFactor[2])

        self.Down_Net_4 = CrossLowR_Net(True, self.Code, self.Channel_mid * self.ChannelScaleFactor[2],
                                        self.Adj_Cr_channels_Low)
        self.DownSample_Net_3 = Downsample(self.Channel_mid * self.ChannelScaleFactor[2], True,
                                           dims=2, out_channels=self.Channel_mid * self.ChannelScaleFactor[2])
        self.input_blocks = nn.ModuleList([self.Down_Net_1,
                                           self.Down_Net_2,self.DownSample_Net_1,
                                           self.Down_Net_3,self.DownSample_Net_2,
                                           self.Down_Net_4,self.DownSample_Net_3])

        # For Mid block
        self.Mid_Net_1 = CrossLowR_Net(True, self.Code, self.Channel_mid *
                                       self.ChannelScaleFactor[2], self.Adj_Cr_channels_Low)
        self.Mid_Net_2 = CrossLowR_Net(True, self.Code, self.Channel_mid *
                                       self.ChannelScaleFactor[2], self.Adj_Cr_channels_Low)
        self.mid_blocks = nn.ModuleList([self.Mid_Net_1,
                                         self.Mid_Net_2])

        # For Up block
        #else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
        self.UpSample_Net_1 = Upsample(self.Channel_mid ** self.ChannelScaleFactor[2], True,
                                           dims=2, out_channels=self.Channel_mid * self.ChannelScaleFactor[2])
        self.Up_Net_1 = CrossLowR_Net(True, self.Code, self.Channel_mid*self.ChannelScaleFactor[2],
                                      self.Adj_Cr_channels_Low)

        self.UpSample_Net_2 = Upsample(self.Channel_mid ** self.ChannelScaleFactor[2], True,
                                       dims=2, out_channels=self.Channel_mid * self.ChannelScaleFactor[1])
        self.Up_Net_2 = CrossLowR_Net(True, self.Code, self.Channel_mid*self.ChannelScaleFactor[1],
                                      self.Adj_Cr_channels_Low)

        self.UpSample_Net_3 = Upsample(self.Channel_mid ** self.ChannelScaleFactor[1], True,
                                       dims=2, out_channels=self.Channel_mid * self.ChannelScaleFactor[0])
        self.Up_Net_3 = CrossLowR_Net(True, self.Code, self.Channel_mid*self.ChannelScaleFactor[0],
                                      self.Adj_Cr_channels_Low)
        self.Up_Net_4 = conv_nd(2, self.Channel_mid, self.Channel_in, 3, padding=1)
        self.output_blocks = nn.ModuleList([self.Up_Net_1,self.UpSample_Net_1,
                                            self.Up_Net_2,self.UpSample_Net_2,
                                            self.Up_Net_3,self.UpSample_Net_3,
                                            self.Up_Net_4])

    def forward(self,x, code):
        mask = []
        x_str=[x]
        # for input blocks
        for i in range(7):
            if i % 2 == 1:
                x, mask_i = self.input_blocks[i](x,code)
                mask.append(mask_i)
                x_str.append(x)
            else:
                x = self.input_blocks[i](x)
                x_str.append(x)

        # for mid blocks
        for i in range(2):
            x, mask_i = self.input_blocks[i](x,code)
            mask.append(mask_i)
            x_str.append(x)

        # for up blocks
        for i in range(7):
            if i % 2 == 0:
                if i != 6:
                    x, mask_i = self.input_blocks[i](x, code)
                    mask.append(mask_i)
                    x_str.append(x)
                else:
                    x = self.input_blocks[i](x)
                    x_str.append(x)
            else:
                x = self.input_blocks[i](x)
                x_str.append(x)
        return mask, x_str

    def Mask_loss(self, mask):
        loss = 0
        l2_loss_fn = nn.MSELoss()
        for i in range(7+2+7):
            loss += l2_loss_fn(mask,mask)
        return loss

    def CodeAcc_feature_loss(self, x_str, code, Decoder_Net_list):
        loss = 0
        BCE_loss_fn = torch.nn.BCELoss()
        for i in range(8):
            if i<4:
                C_decoded = Decoder_Net_list[i].forward(x_str[i])
                loss+= BCE_loss_fn(code,C_decoded)
            else:
                C_decoded = Decoder_Net_list[7-i].forward(x_str[i])
                loss += BCE_loss_fn(code, C_decoded)
        return loss

    def CodeAcc_loss(self,x,code, Decoder_Net):
        C_decoded = Decoder_Net.forward(x)
        BCE_loss_fn = torch.nn.BCELoss()
        return BCE_loss_fn(code, C_decoded)

    def Acc_Code_eval(self,x,code, Decoder_Net):
        c_d=Decoder_Net.forward(x)
        count = 0
        for i in range(len(code)):
            if code[i] == c_d[i]:
                count+=1
        return count/len(code)

    def recon_loss(self,x_str):
        # for latter, use LPIPS. Here use this: SSIM
        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        loss = 1 - ssim_loss(x_str[0],x_str[-1])
        return loss

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out, eps=1e-3),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)

class Decoder_Net(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    I just got this from stable signature
    """
    def __init__(self, num_blocks, num_bits, channels, redundancy=1):

        super(Decoder_Net, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits*redundancy))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits*redundancy, num_bits*redundancy)

        self.num_bits = num_bits
        self.redundancy = redundancy

    def forward(self, img_w):

        x = self.layers(img_w) # b d 1 1
        x = x.squeeze(-1).squeeze(-1) # b d
        x = self.linear(x)

        x = x.view(-1, self.num_bits, self.redundancy) # b k*r -> b k r
        x = torch.sum(x, dim=-1) # b k r -> b k

        return x






shape = [4, 256//8, 256//8]
C, H, W = shape
batch_size = 1
size = (batch_size, C, H, W)
img = torch.randn(size)
code ="0010011"
TestCodeNet=CodeEmb_Net(code)
TestCodeNet.forward()
TestNet=CrossLowR_Net(use_conv=True, Code=code, Cr_channels_in = 4, Adj_Cr_channels_Low = 2,
                      Adj_Cr_n_heads=4, Adj_Cr_d_head=4, dims=2, padding='same')
TestNet.forward(img,code)