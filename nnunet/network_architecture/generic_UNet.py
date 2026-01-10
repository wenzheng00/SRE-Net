#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional

from mamba_ssm import Mamba
import torch.nn.functional as F 
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
import math
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

class Convs(torch.nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride_size, num_convs=2) -> None:
        super().__init__()
        self.conv_layer = StackedConvBlocks(
                num_convs=num_convs,
                conv_op=nn.Conv3d,
                input_channels=in_chans,
                output_channels=out_chans,
                kernel_size=kernel_size,
                initial_stride=stride_size,
                conv_bias=True,
                norm_op=nn.InstanceNorm3d,
                norm_op_kwargs={'eps': 1e-5, 'affine': True},
                dropout_op=None,
                dropout_op_kwargs=None,
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={"inplace": True},
                nonlin_first=False
        )
    def forward(self, x):
        return self.conv_layer(x)

class MambaLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.InstanceNorm3d(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                #bimamba_type="v3",
        )
        
        self.mlp_norm = nn.InstanceNorm3d(dim)
        self.mlp = MlpChannel(dim, 4 * dim)
        self.ema = SPPE(dim)
    
    def forward(self, x):
        x_skip = x
        y = self.ema(x)
        x = self.norm(x)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x = self.mamba(x)

        x = x.transpose(-1, -2).reshape(B, C, *img_dims)

        x = self.mlp_norm(x)
        x = x * y
        x = self.mlp(x) 

        return x + x_skip
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[16, 32, 64, 128]):
        super().__init__()
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              Convs(dims[0], dims[0], kernel_size=3, stride_size=1, num_convs=2),
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        # self.gscs = nn.ModuleList()
        for i in range(4):
            if i == 0 or i == 1:
                stage = Convs(in_chans=dims[i], out_chans=dims[i], 
                              kernel_size=3, stride_size=1, num_convs=3)
            else :
                stage = nn.Sequential(
                *[MambaLayer(dim=dims[i]) for j in range(2)]
                )
                # stage = Convs(in_chans=dims[i], out_chans=dims[i], 
                #               kernel_size=3, stride_size=1, num_convs=3)
            
            self.stages.append(stage)         

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class SPPE(nn.Module):
    def __init__(self, channels, factor=8):
        super(SPPE, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_d = nn.AdaptiveAvgPool3d((1, 1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv3d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv3d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w, d = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w, d)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2, 4)
        x_d = self.pool_d(group_x).permute(0, 1, 4, 3, 2)
        hwd = self.conv1x1(torch.cat([x_h, x_w, x_d], dim=2))
        x_h, x_w, x_d = torch.split(hwd, [h, w, d], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2, 4).sigmoid() * x_d.permute(0, 1, 4, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w, d)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w, d)

class GroupBatchnorm3d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm3d, self).__init__()  
        assert c_num >= group_num  
        self.group_num = group_num  
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1, 1))  
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1, 1))  
        self.eps = eps 

    def forward(self, x):
        N, C, H, W, D = x.size()  
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True) 
        std = x.std(dim=2, keepdim=True) 
        x = (x - mean) / (std + self.eps)  
        x = x.view(N, C, H, W, D)  
        return x * self.gamma + self.beta  
    
class SAR(nn.Module):
    def __init__(self,
                 oup_channels: int,  
                 group_num: int = 16,  
                 gate_treshold: float = 0.5,  
                 torch_gn: bool = False  
                 ):
        super().__init__()  

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm3d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold  
        self.sigomid = nn.Sigmoid()  
        self.mhda = MultiHeadDiffAttention(256)

    def forward(self, x):
        gn_x = self.gn(x)  
        w_gamma = self.gn.gamma / sum(self.gn.gamma) 
        reweights = self.sigomid(gn_x * w_gamma)  

        info_mask = reweights >= self.gate_treshold  
        noninfo_mask = reweights < self.gate_treshold 
        x_1 = info_mask * x  
        x_1 = self.mhda(x_1)
        x_2 = noninfo_mask * x  
        x = self.reconstruct(x_1, x_2)  
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)  
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)  
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)  

class CAR(nn.Module):
    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2,
                 group_kernel_size: int = 3):
        super().__init__() 

        self.up_channel = up_channel = int(alpha * op_channel) 
        self.low_channel = low_channel = op_channel - up_channel 
        self.squeeze1 = nn.Conv3d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)  
        self.squeeze2 = nn.Conv3d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)  

        self.PWC2 = nn.Conv3d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)  
        self.advavg = nn.AdaptiveAvgPool3d(1)  
        self.mhda = MultiHeadDiffAttention(64)
        self.conv = nn.Conv3d(64, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        #Y1 = self.GWC(up) + self.PWC1(up)
        Y1 = self.mhda(up)
        Y1 = self.conv(Y1)

        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2

class SCAR(nn.Module):
    def __init__(self, op_channel: int, group_num: int = 16, gate_treshold: float = 0.5, alpha: float = 1 / 2,
                 squeeze_radio: int = 2, group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()  

        self.sar = SAR(op_channel, group_num=group_num, gate_treshold=gate_treshold) 
        self.car = CAR(op_channel, alpha=alpha, squeeze_radio=squeeze_radio, group_size=group_size,
                       group_kernel_size=group_kernel_size)  

    def forward(self, x):
        x = self.sar(x)  
        x = self.car(x)  
        return x
    
def lambda_init(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * (depth - 1))

# Multi-head Differential Attention
class MultiHeadDiffAttention(nn.Module):
    def __init__(self, n_embd, n_head=4, layer_idx=4, dropout=0.2):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.lambda_init = lambda_init(layer_idx) 

        # split qkv
        self.q1_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.q2_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k1_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k2_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, 2 * n_embd, bias=False)  # V projects to 2 * n_embd

        self.c_proj = nn.Linear(2 * n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.subln = nn.LayerNorm(2 * self.head_size, elementwise_affine=False)

        # Init λ across heads
        self.lambda_q1 = nn.Parameter(torch.randn(n_head, self.head_size) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(n_head, self.head_size) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(n_head, self.head_size) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(n_head, self.head_size) * 0.1)

    def forward(self, x):
        B, C, H, W, D = x.shape
        x = x.reshape(B, H*W*D, C)
        B, T, C = x.shape

        # Project x to get q1, q2, k1, k2, v
        q1 = self.q1_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q2 = self.q2_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k1 = self.k1_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k2 = self.k2_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, 2 * self.head_size).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_size)
        att1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        att2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        attn_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att1 = att1.masked_fill(attn_mask == 0, float('-inf'))
        att2 = att2.masked_fill(attn_mask == 0, float('-inf'))

        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)

        # Compute λ for each head separately
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        att = att1 - lambda_full * att2
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)  # [B, n_head, T, 2 * head_size]
        y = self.subln(y)
        y = y * (1 - self.lambda_init)

        y = y.transpose(1, 2).contiguous().view(B, T, 2 * C)
        y = self.resid_dropout(self.c_proj(y))
        y = y.reshape(B, C, H, W, D)
        return y
    
class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = nn.Conv3d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    

class EIE(nn.Module):
    def __init__(self, in_channels):
        super(EIE, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 3 , 1, 1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm3d(1),
            nn.Sigmoid())

        #self.cbam = CEA(in_channels)
        self.conv = nn.Conv3d(2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self,  edge_feature, x, pred):
        residual = x#[1,64,128,128,128]
        xsize = x.size()[2:]#[128,128,128]
        pred = self.conv(pred)
        pred = torch.sigmoid(pred)#[1,1,128,128,128]
        
        #reverse attention 
        background_att = 1 - pred#[1,1,128,128,128]
        background_x= x * background_att#[1,64,128,128,128]
        
        #boudary attention
        # edge_pred = make_laplace(pred, 1)  #[1,1,128,128,128]
        # pred_feature = x * edge_pred#[1,64,128,128,128]

        #high-frequency feature
        edge_input = F.interpolate(edge_feature, size=xsize, mode='trilinear', align_corners=True)#[1,1,128,128,128]
        input_feature = x * edge_input#[1,64,128,128,128]

        fusion_feature = torch.cat([background_x, input_feature], dim=1)#[1,128,128,128,128]
        fusion_feature = self.fusion_conv(fusion_feature)#[1,64,128,128,128]

        attention_map = self.attention(fusion_feature)#[1,1,128,128,128]
        fusion_feature = fusion_feature * attention_map#[1,64,128,128,128]

        out = fusion_feature + residual#[1,64,128,128,128]
        #out = self.cbam(out)#[1,64,128,128,128]
        return out
    
def laplacian_pyramid(x, num_levels=3):
        pyramid = []
        current_level = x
    
    # 构建高斯金字塔，并计算每一层的拉普拉斯金字塔
        for i in range(num_levels):
            # 对当前层进行高斯模糊，生成下一层
            blurred = F.avg_pool3d(current_level, 2, stride=2)  # 下采样
            pyramid.append(current_level - F.interpolate(blurred, scale_factor=2, mode='trilinear', align_corners=False))
            current_level = blurred
        
        # 将金字塔中的所有层返回
        pyramid.append(current_level)  # 最后一层没有高斯层用于计算差异
        return pyramid

# 恢复拉普拉斯金字塔
def reconstruct_from_pyramid(pyramid):
        x_reconstructed = pyramid[-1]  # 从最底层开始
        
        for i in range(len(pyramid) - 2, -1, -1):
            # 对每一层进行插值恢复并加回拉普拉斯差异
            x_reconstructed = F.interpolate(x_reconstructed, scale_factor=2, mode='trilinear', align_corners=False)
            x_reconstructed = x_reconstructed + pyramid[i]
        
        return x_reconstructed
class DConv_3x3x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation, d=1):
        super(DConv_3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 1), stride=1, padding=(d, d, 0), dilation=d, bias=True)
        self.norm = nn.InstanceNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x
class Conv_1x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.norm = nn.InstanceNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x
class Conv_1x1x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x1x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.norm = nn.InstanceNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x
    
class DHDC_module(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(DHDC_module, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = in_dim // 4
        self.out_inter_dim = out_dim // 4
        self.conv_3x3x1_1 = DConv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation, d=1)
        self.conv_3x3x1_2 = DConv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation, d=2)
        self.conv_3x3x1_3 = DConv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation, d=3)
        self.conv_1x1x1_1 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x1x1_2 = Conv_1x1x1(out_dim, out_dim, activation)
        if self.in_dim > self.out_dim:
            self.conv_1x1x1_3 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x3x3 = Conv_1x3x3(out_dim, out_dim, activation)

    def forward(self, x):
        x_1 = self.conv_1x1x1_1(x)
        x1 = x_1[:, 0:self.out_inter_dim, ...]
        x2 = x_1[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        x3 = x_1[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        x4 = x_1[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]
        s2 = self.conv_3x3x1_1(x2)
        z2 = x2 + s2
        s3 = self.conv_3x3x1_2(s2 + x3)
        z3 = x3 + s3
        s4 = self.conv_3x3x1_3(s3 + x4)
        z4 = x4 + s4
        x_1 = torch.cat((x1, z2, z3, z4), dim=1)
        x_1 = self.conv_1x1x1_2(x_1)
        if self.in_dim > self.out_dim:
            x = self.conv_1x1x1_3(x)
        x_1 = self.conv_1x3x3(x + x_1)
        return x_1
    
def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm3d(out_dim),
        activation)
class HDC_module(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(HDC_module, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = in_dim // 4
        self.out_inter_dim = out_dim // 4
        self.conv_3x3x1_1 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_3x3x1_2 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_3x3x1_3 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_1x1x1_1 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x1x1_2 = Conv_1x1x1(out_dim, out_dim, activation)
        if self.in_dim > self.out_dim:
            self.conv_1x1x1_3 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x3x3 = Conv_1x3x3(out_dim, out_dim, activation)

    def forward(self, x):
        x_1 = self.conv_1x1x1_1(x)
        x1 = x_1[:, 0:self.out_inter_dim, ...]
        x2 = x_1[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        x3 = x_1[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        x4 = x_1[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]
        x2 = self.conv_3x3x1_1(x2)
        x3 = self.conv_3x3x1_2(x2 + x3)
        x4 = self.conv_3x3x1_3(x3 + x4)
        x_1 = torch.cat((x1, x2, x3, x4), dim=1)
        x_1 = self.conv_1x1x1_2(x_1)
        if self.in_dim > self.out_dim:
            x = self.conv_1x1x1_3(x)
        x_1 = self.conv_1x3x3(x + x_1)
        return x_1
class Conv_3x3x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=True)
        self.norm = nn.InstanceNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x
    
class Generic_UNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, 
                 #basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False,
                 in_chans=1,  
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[16, 32, 64, 128],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 256,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        upsample_mode = 'trilinear'
        self.output_features = base_num_features
        self.input_features = input_channels
        self.num_pool = num_pool
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.activation = nn.ReLU(inplace=False)

        #添加segmamba
        self.vit = MambaEncoder(in_chans, 
                                depths=depths,
                                dims=feat_size,        
                              )
 
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.scar = SCAR(256)
        
        self.eie2 = EIE(16)
        self.eie3 = EIE(32)
        self.eie4 = EIE(64)
        self.eie5 = EIE(128)


        self.out5 = Out(256, 2)
        self.out4 = Out(128, 2)
        self.out3 = Out(64, 2)
        self.out2 = Out(32, 2)
        self.out1 = Out(16, 2)
        #self.conv2 = nn.Conv3d(4, 2, kernel_size=1, stride=1, padding=0)

        self.conv_4 = DHDC_module(384, 128, self.activation)
        self.up_4 = conv_trans_block_3d(128, 128, self.activation)
        self.conv_3 = DHDC_module(192, 64, self.activation)
        self.up_3 = conv_trans_block_3d(64, 64, self.activation)
        self.conv_2 = HDC_module(96, 32, self.activation)
        self.up_2 = conv_trans_block_3d(32, 32, self.activation)
        self.conv_1 = HDC_module(48, 16, self.activation)
        #self.up_1 = conv_trans_block_3d(16, 16, self.activation)

        self.up_5 = conv_trans_block_3d(256, 256, self.activation)

    def forward(self, x):
        #print("开始了")
        # 构建拉普拉斯金字塔
        pyramid = laplacian_pyramid(x, num_levels=3)
        # 重建回原始形状
        edge_feature = reconstruct_from_pyramid(pyramid)#[2,1,128,128,128]

        skips = []
        seg_outputs = []
        #segmamba
        #encoder
        outs = self.vit(x)
        enc1 = self.encoder1(x)#[2,16,128,128,128]
        x2 = outs[0]
        enc2 = self.encoder2(x2)#[2,32,64,64,64]
        x3 = outs[1]
        enc3 = self.encoder3(x3)#[2,64,32,32,32]
        x4 = outs[2]
        enc4 = self.encoder4(x4)#[2,128,16,16,16]
        enc_hidden = self.encoder5(outs[3])#[2,256,8,8,8]
        enc_hidden = self.scar(enc_hidden)#[2,256,8,8,8]

        d5 = self.up_5(enc_hidden)#[2,256,16,16,16]
        #d5 = self.up5(enc_hidden)#[2,256,16,16,16]
        out5 = self.out5(d5)#[2,2,16,16,16]
        eie5 = self.eie5(edge_feature, enc4, out5)#[2,128,16,16,16]

        d4 = torch.cat((d5, eie5), dim=1)#[2,384,16,16,16]
        d4 = self.conv_4(d4)#[2,128,16,16,16]
        d4 = self.up_4(d4)#[2,128,32,32,32]
        #d4 = self.up4(d5,ega5)#[2,128,32,32,32]

        out4 = self.out4(d4)#[2,2,32,32,32]
        eie4 = self.eie4(edge_feature, enc3, out4)#[2,64,32,32,32]

        d3 = torch.cat((d4, eie4), dim=1)
        d3 = self.conv_3(d3)
        d3 = self.up_3(d3)#[2,64,64,64,64]
        #d3 = self.up3(d4,ega4)#[2,64,64,64,64]

        out3 = self.out3(d3)#[2,2,64,64,64]
        eie3 = self.eie3(edge_feature, enc2, out3)#[2,32,64,64,64]

        d2 = torch.cat((d3, eie3), dim=1)
        d2 = self.conv_2(d2)
        d2 = self.up_2(d2)#[2,32,128,128,128]
        #d2 = self.up2(d3,ega3)#[2,32,128,128,128]

        out2 = self.out2(d2)#[2,2,128,128,128]
        eie2 = self.eie2(edge_feature, enc1, out2)#[2,16,128,128,128]

        d1 = torch.cat((d2, eie2), dim=1)
        d1 = self.conv_1(d1)

        out = self.out1(d1)#[2,2,128,128,128]
        
        seg_outputs=[out]
        
        if self._deep_supervision and self.do_ds:
            return seg_outputs
        else:
            return seg_outputs[0]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
