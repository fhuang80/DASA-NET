import torch
import sys
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..layers import (gather_operation, grouping_operation, three_nn, three_interpolate)

import faiss  # 导入 faiss 库用于最近邻计算

"""
标准空间卷积假设输入数据具有规则的邻域结构。现有方法通常通过例如固定邻域大小来固定规则“视图”，从而将卷积推广到不规则点云域，其中每个点的卷积核大小保持相同。
然而，由于点云不像图像那样结构化，固定的邻居数会产生不幸的归纳偏差。我们提出了一种新颖的图卷积，称为差异图卷积（diffConv），它不依赖于常规视图。
diffConv 在空间变化和密度扩张的邻域上运行，这些邻域通过学习的屏蔽注意力机制进一步适应。
实验表明，我们的模型对噪声非常鲁棒，在 3D 形状分类和场景理解任务中获得了最先进的性能，并且推理速度更快。
"""

def get_dist(src, dst):
    B, _, N = src.shape
    _, _, M = dst.shape
    dist = -2 * torch.matmul(src.transpose(1, 2), dst)  # [B, N, M]
    dist += torch.sum(src ** 2, dim=1).view(B, N, 1)  # [B, N, 1]
    dist += torch.sum(dst ** 2, dim=1).view(B, 1, M)  # [B, 1, M]
    return dist

def index_points(points, idx):
    new_points = gather_operation(points, idx)  # gather_operation 默认处理 [B, C, N]
    return new_points

def sample_and_group(radius, k, xyz, feat, centroid, dist):
    device = xyz.device
    B, _, N = xyz.shape
    _, _, M = centroid.shape

    idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, M, 1])

    idx[dist > radius] = N
    idx = idx.sort(dim=-1)[0][:, :, :k]
    group_first = idx[:, :, 0].view(B, M, 1).repeat([1, 1, k])
    mask = (idx == N)
    idx[mask] = group_first[mask]

    torch.cuda.empty_cache()
    idx = idx.int().contiguous()

    cent_feat = grouping_operation(feat, idx)  # grouping 操作直接处理 [B, C, N]
    return cent_feat, idx


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, act=nn.GELU(), bias_=False):
        super(Conv1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias_),
            nn.BatchNorm1d(out_channels)
        )
        self.act = act
        nn.init.xavier_normal_(self.conv[0].weight.data)

    def forward(self, x):
        x = self.conv(x)  # [B, C, N] -> [B, C_out, N]
        if self.act is not None:
            return self.act(x)
        else:
            return x

class PositionEncoder(nn.Module):
    def __init__(self, out_channel, radius, k=20):
        super(PositionEncoder, self).__init__()
        self.k = k
        self.xyz2feature = nn.Sequential(
            nn.Conv2d(9, out_channel // 8, kernel_size=1),
            nn.BatchNorm2d(out_channel // 8),
            nn.GELU()
        )
        self.mlp = nn.Sequential(
            Conv1x1(out_channel // 8, out_channel // 4),
            Conv1x1(out_channel // 4, out_channel, act=None)
        )

    def forward(self, centroid, xyz, radius, dist):
        point_feature, _ = sample_and_group(radius, self.k, xyz, xyz, centroid, dist)  # [B, C, N, k]
        points = centroid.unsqueeze(3).repeat(1, 1, 1, self.k)  # [B, 3, N, k]
        variance = point_feature - points
        point_feature = torch.cat((points, point_feature, variance), dim=1)  # [B, 9, N, k]
        point_feature = self.xyz2feature(point_feature)  # [B, out_channel//8, N, k]
        point_feature = torch.max(point_feature, dim=-1)[0]  # [B, out_channel//8, N]
        point_feature = self.mlp(point_feature)  # [B, out_channel, N]
        return point_feature

class MaskedAttention(nn.Module):
    def __init__(self, in_channels, hid_channels=128):
        super().__init__()
        if not hid_channels or hid_channels < 1:
            hid_channels = 1

        self.conv_q = Conv1x1(in_channels + 3, hid_channels, act=None)
        self.conv_k = Conv1x1(in_channels + 3, hid_channels, act=None)

    def forward(self, cent_feat, feat, mask):
        q = self.conv_q(cent_feat)  # [B, M, in_channels+3] -> [B, M, C_int]
        k = self.conv_k(feat)  # [B, N, in_channels+3] -> [B, N, C_int]

        adj = torch.bmm(q, k.transpose(1, 2))  # [B, M, C_int] * [B, C_int, N] -> [B, M, N]

        adj = adj.masked_fill(mask < 1e-9, -1e9)
        adj = torch.softmax(adj, dim=-1)
        adj = F.normalize(adj, p=1, dim=1)
        adj = F.normalize(adj, p=1, dim=-1)

        return adj

def dilated_ball_query(dist, h, base_radius, max_radius):
    sigma = 1
    gauss = torch.exp(-(dist) / (2 * (h ** 2) * (sigma ** 2)))
    kd_dist = torch.sum(gauss, dim=-1).unsqueeze(-1)
    kd_score = kd_dist / (torch.max(kd_dist, dim=1)[0].unsqueeze(-1) + 1e-9)
    radius = base_radius + (max_radius - base_radius) * kd_score
    return radius

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g = g.transpose(1, 2)
        x = x.transpose(1, 2)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.gelu(g1 + x1)
        psi = self.psi(psi)
        psi = psi.transpose(1, 2)

        return psi

class PointFeaturePropagation(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        super(PointFeaturePropagation, self).__init__()
        in_channel = in_channel1 + in_channel2
        self.conv = nn.Sequential(
            Conv1x1(in_channel, in_channel // 2),
            Conv1x1(in_channel // 2, in_channel // 2),
            Conv1x1(in_channel // 2, out_channel)
        )
        self.att = Attention_block(in_channel1, in_channel2, in_channel2)

    def forward(self, xyz1, xyz2, feat1, feat2):
        dists, idx = three_nn(xyz1, xyz2)
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm
        int_feat = three_interpolate(feat2.transpose(1, 2).contiguous(), idx, weight).transpose(1, 2)

        psix = self.att(int_feat, feat1)
        feat1 = feat1 * psix

        if feat1 is not None:
            cat_feat = torch.cat((feat1, int_feat), dim=-1)
        else:
            cat_feat = int_feat
        cat_feat = self.conv(cat_feat)

        return cat_feat

class HeadDiffConv(nn.Module):
    def __init__(self, in_channels, out_channels, base_radius, bottleneck=4):
        super(HeadDiffConv, self).__init__()
        self.conv_v = Conv1x1(2 * in_channels, out_channels, act=None)
        self.mat = MaskedAttention(in_channels, in_channels // bottleneck)
        self.pos_conv = PositionEncoder(out_channels, np.sqrt(base_radius))
        self.base_radius = base_radius
        self.output_conv = Conv1x1(out_channels, 32, act=None)

    def forward(self, xyz):
        batch_size, _, point_num = xyz.size()

        centroid = xyz.clone()
        x = xyz

        dist = get_dist(centroid, xyz)

        radius = dilated_ball_query(dist, h=0.1, base_radius=self.base_radius, max_radius=self.base_radius * 2)
        mask = (dist < radius).float()

        emb_cent = torch.cat((x, centroid), dim=1)
        emb_x = torch.cat((x, xyz), dim=1)
        adj = self.mat(emb_cent, emb_x, mask)

        smoothed_x = torch.bmm(x, adj)
        variation = smoothed_x - x

        x = torch.cat((variation, x), dim=1)
        x = self.conv_v(x)

        pos_emb = self.pos_conv(centroid, xyz, radius, dist)
        x = x + pos_emb
        x = F.gelu(x)

        x = self.output_conv(x)

        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pc = torch.rand(32, 3, 1024).to(device)

    head_diffconv = HeadDiffConv(3, 32, 0.2).to(device)

    new_feat_head = head_diffconv(pc)

    print("Head layer output feature shape:", new_feat_head.shape)  # 期望输出 [32, 32, 1024]