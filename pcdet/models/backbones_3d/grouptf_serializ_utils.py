from functools import partial

import math
import numpy as np
import torch
import torch.nn as nn
import torch_scatter
from PIL.ImageChops import offset
from mamba_ssm import Block as MambaBlock

from pcdet.ops.pointnet2.pointnet2_stack.voxel_query_utils import voxel_query
from torch.nn import functional as F

from ..model_utils.retnet_attn import Block as RetNetBlock
from ..model_utils.rwkv_cls import Block as RWKVBlock
from ..model_utils.vision_lstm2 import xLSTM_Block
from ..model_utils.ttt import TTTBlock
from ...utils.spconv_utils import replace_feature, spconv
import torch.utils.checkpoint as cp
# from ...lib.pointops.functions import pointops
from ...ops.pointnet2.pointnet2_batch import pointnet2_utils
from ...pointops.functions.pointops import KNNQuery
from .hilbert import encode as hilbert_encode_
from .z_order import xyz2key as z_order_encode_

# from .show_points import show_voxel

knnquery = KNNQuery.apply


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z", num_dims=3):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        if num_dims == 3:
            code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
        elif num_dims == 2:
            code = z_order_encode(grid_coord[:, [1, 0]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth, num_dims=num_dims)
    elif order == "hilbert-trans":
        if num_dims == 3:
            code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth, num_dims=num_dims)
        elif num_dims == 2:
            code = hilbert_encode(grid_coord[:, [1, 0]], depth=depth, num_dims=num_dims)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code

    # order = torch.argsort(code)
    # inverse = torch.zeros_like(order).scatter_(
    #     dim=1,
    #     index=order,
    #     src=torch.arange(0, code.shape[1], device=order.device).repeat(
    #         code.shape[0], 1
    #     ),
    # )
    return code


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16, num_dims=3):
    return hilbert_encode_(grid_coord, num_dims=num_dims, num_bits=depth)


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            # nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            # nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1),
            nn.Linear(num_pos_feats, num_pos_feats),
        )

    def forward(self, xyz):
        # xyz = xyz.transpose(1, 2).contiguous()
        xyz = xyz[:, 1:].contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding.contiguous()


class LBPEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288, nsample=16):
        super().__init__()
        self.nsample = nsample
        self.position_embedding_head = nn.Sequential(
            nn.Linear((self.nsample - 1) * input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats),
        )

    def forward(self, x):
        o = []
        count = 0
        for i in range(x.batch_size):
            count += x.indices[:, 0][x.indices[:, 0] == i].shape[0]
            o.append(count)
        o = torch.cuda.IntTensor(o)
        idx, diff = knnquery(self.nsample, x.indices[:, 1:].float(), x.indices[:, 1:].float(), o, o)
        x_indices_n = x.indices[:, 1:][[idx]]
        x_indices_pos = (x_indices_n[:, 1:, :] - x_indices_n[:, 0:1, :]).reshape(-1, (self.nsample - 1) * 3)
        position_embedding = self.position_embedding_head(x_indices_pos.float())
        return position_embedding


class my_MultiheadAttention(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, d_model, nhead, group_size, attn_drop):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.group_size = group_size
        self.softmax = torch.nn.Softmax(dim=-1)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.scale = (d_model // nhead) ** -0.5

    def forward(self, q, k, v, mask):
        H = self.nhead
        G = self.group_size
        C = self.d_model
        q = q.reshape(-1, G, H, C // H).transpose(2, 1)  # (N', H, G, C')
        k = k.reshape(-1, G, H, C // H).transpose(2, 1)  # (N', H, G, C')
        v = v.reshape(-1, G, H, C // H).transpose(2, 1)  # (N', H, G, C')
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, G, G)

        attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(q.dtype)
        feat = (attn @ v).transpose(1, 2).reshape(-1, G, C)
        return feat, attn


class self_att_Layer(nn.Module):
    def __init__(self, d_model, group_size, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.group_size = group_size
        # self.self_posembed = PositionEmbeddingLearned(3, d_model)

        # self.self_posembed = LBPEmbeddingLearned(3, d_model, 8)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # my self_att
        self.linear_qkv = nn.Linear(d_model, d_model * 3)
        self.self_attn = my_MultiheadAttention(d_model, nhead, group_size, dropout)

        # official self_att
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    # def with_pos_embed(self, tensor, pos_embed):
    #     return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, x_features, x_coords, mask):
        # NxCxP to PxNxC
        # if self.self_posembed is not None:
        #     x_pos_embed = self.self_posembed(x_coords.float())
        # else:
        #     x_pos_embed = None
        # x_pos_embed = self.self_posembed(x_coords).float()
        # x_features = x_features + x_pos_embed

        # my self_att
        qkv = self.linear_qkv(x_features).reshape(-1, self.group_size, 3, self.d_model)
        query, key, value = qkv.unbind(2)
        query, _ = self.self_attn(q=query, k=key, v=value, mask=mask)

        # official self_att
        # query = self.self_attn(q = x_features, k = x_features, v=x_features)[0]

        ####
        query = x_features + self.dropout1(query)
        query = self.norm1(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        return query


class cross_att_Layer(nn.Module):
    def __init__(self, d_model, group_size, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.group_size = group_size
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_kv = nn.Linear(d_model, d_model * 2)

        self.cross_attn = my_MultiheadAttention(d_model, nhead, group_size, dropout)

        ########################
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        self.norm2 = nn.LayerNorm(d_model)

    ########################
    # self.dropout = nn.Dropout(dropout)
    # self.MLP1 = nn.Sequential( # nn.Linear(in_features=d_model, out_features=d_model),
    #                          nn.LayerNorm(d_model),
    #                          nn.LeakyReLU(negative_slope=0.2))
    # self.MLP2 = nn.Sequential(nn.Linear(in_features=d_model, out_features=d_model),
    #                           nn.LayerNorm(d_model),
    #                           nn.LeakyReLU(negative_slope=0.2))

    def forward(self, query_features, KV_features, mask):  # x_features:[N', G, C']

        query = self.linear_q(query_features).reshape(-1, self.group_size, self.d_model)
        key, value = self.linear_kv(KV_features).reshape(-1, self.group_size, 2, self.d_model).unbind(2)

        query, _ = self.cross_attn(q=query, k=key, v=value, mask=mask)

        ##########################
        query = query_features + self.dropout1(query)
        query = self.norm1(query)
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        ##########################
        # hidden_features = self.MLP1(self.dropout(query))
        # error_features = self.MLP2(hidden_features - query_features)
        # query = query_features + error_features

        return query


# @torch.inference_mode()
def index_to_group(group_size, index_win_X, window_num_zyx, batch_size, device, index_in_win=None):
    countnum_voxel_in_win_X_flat = torch.bincount(index_win_X)  # 计算每个win有多少voxel
    zeros_tensor_X = torch.zeros(window_num_zyx * batch_size - countnum_voxel_in_win_X_flat.size()[0]).int().to(device)
    countnum_voxel_in_win_X_flat = torch.cat((countnum_voxel_in_win_X_flat, zeros_tensor_X))  # 计算每个win有多少voxel，补满尾部
    countnum_group_in_win_X_flat = -(-countnum_voxel_in_win_X_flat // group_size)  # 计算每个win有多少group
    voxel_missing_group_in_win_X_flat = (
            countnum_group_in_win_X_flat * group_size - countnum_voxel_in_win_X_flat).int()  # 计算每个win缺少几个voxel
    countnum_group_in_win_X_flat_cumsum = torch.cumsum(countnum_group_in_win_X_flat, 0)  # 每个win在第几个group结束
    sorted_index_win_X_flat, raw2sorted_indices_index_win_X = torch.sort(index_win_X, dim=0)  # voxel所在第几个win的排序

    # voxel在第几个win,排序
    index_voxel_in_win_X_unique = torch.unique(sorted_index_win_X_flat).int()

    # if sorted_index_win_X_flat[0]:
    #     sorted_countnum_group_in_win_X_flat = countnum_group_in_win_X_flat_cumsum[sorted_index_win_X_flat-1].int()        # voxel对应的窗口在第几个group(开始/结束)
    # else:
    #     sorted_countnum_group_in_win_X_flat = torch.zeros(sorted_index_win_X_flat.size()[0]).int().to(device)
    #     sorted_countnum_group_in_win_X_flat[sorted_index_win_X_flat!=0] = countnum_group_in_win_X_flat_cumsum[sorted_index_win_X_flat[sorted_index_win_X_flat!=0] - 1].int()

    sorted_countnum_group_in_win_X_flat = torch.zeros(sorted_index_win_X_flat.size()[0]).int().to(device)
    sorted_countnum_group_in_win_X_flat[sorted_index_win_X_flat != 0] = countnum_group_in_win_X_flat_cumsum[
        sorted_index_win_X_flat[sorted_index_win_X_flat != 0] - 1].int()

    ######################## 空体素全堆最后一个
    # out, inverse_indices, return_counts = torch.unique(sorted_countnum_group_in_win_X_flat, return_inverse=True,
    #                                                    return_counts=True)
    # # return_counts:该win有几个voxel,
    # return_counts = torch.cumsum(return_counts[0:-1], dim=0)
    # addcuncum = torch.ones(sorted_countnum_group_in_win_X_flat.size()[0]).int().to(device)
    # addcuncum[0] = 0
    # addcuncum[return_counts] += voxel_missing_group_in_win_X_flat[index_voxel_in_win_X_unique][0:-1]

    ######################## 空体素平均分配
    num_counts_group = torch.zeros(countnum_group_in_win_X_flat.sum()).int().to(device)
    return_counts_group = torch.zeros(countnum_group_in_win_X_flat.sum()).int().to(device)
    voxel_missing_group = torch.zeros(countnum_group_in_win_X_flat.sum()).int().to(device)
    p = 0
    for i in range(len(countnum_group_in_win_X_flat)):
        lack_voxel = voxel_missing_group_in_win_X_flat[i]
        groupnum = countnum_group_in_win_X_flat[i]
        if groupnum:
            SupplementALL = lack_voxel // groupnum

            SupplementSingle = lack_voxel % groupnum

            num_counts_group[p:p + groupnum] = group_size - SupplementALL
            num_counts_group[p + groupnum - SupplementSingle:p + groupnum] += -1

            voxel_missing_group[p:p + groupnum] = SupplementALL
            voxel_missing_group[p + groupnum - SupplementSingle:p + groupnum] += 1

            p += groupnum
    return_counts_group = torch.cumsum(num_counts_group[0:-1], dim=0)
    addcuncum = torch.ones(sorted_countnum_group_in_win_X_flat.size()[0]).int().to(device)
    addcuncum[0] = 0
    addcuncum[return_counts_group] += voxel_missing_group[0:-1]

    ########################

    sorted_countnum_voxel_in_group_in_win_X_flat = torch.cumsum(addcuncum, dim=0)  # 该voxel在第几个group第几个值的索引，由group使用

    # mappings = {"raw2sorted_indices_index_win_X": raw2sorted_indices_index_win_X, "sorted_countnum_voxel_in_group_in_win_X_flat": sorted_countnum_voxel_in_group_in_win_X_flat}
    # mappings = raw2sorted_indices_index_win_X,sorted_countnum_voxel_in_group_in_win_X_flat
    # index_flat = index[:,0]*XYZ + index[:,1] + index[:,2]*XZ+index[:,3]*Z

    # index_in_win_u_in_sortedwin = index_in_win[raw2sorted_indices_index_win_X]
    # sorted_index_in_win_u_in_sortedwin, raw2sorted_indices_index_in_win_u_in_sortedwin = torch.sort(index_in_win_u_in_sortedwin, dim=0)
    # sorted_index_in_win_u_in_sortedwin2, raw2sorted_indices_index_in_win_u_in_sortedwin2 = torch.sort(index_in_win, dim=0)
    # return raw2sorted_indices_index_win_X[raw2sorted_indices_index_in_win_u_in_sortedwin], sorted_countnum_voxel_in_group_in_win_X_flat, countnum_group_in_win_X_flat.sum()

    return sorted_countnum_voxel_in_group_in_win_X_flat, countnum_group_in_win_X_flat.sum()


def sprace_to_groupfearutes(group_size, window_num, index, batch_size, serializ, sparse_shape, dim, device):
    Win_Z_shape = math.ceil(sparse_shape[0] / window_num[0])
    Win_Y_shape = math.ceil(sparse_shape[1] / window_num[1])
    Win_X_shape = math.ceil(sparse_shape[2] / window_num[2])

    # Win_Z_shape = window_shape[0]
    # Win_Y_shape = window_shape[1]
    # Win_X_shape = window_shape[2]

    window_num = [(-(-sparse_shape[0] // Win_Z_shape)), (-(-sparse_shape[1] // Win_Y_shape)),
                  (-(-sparse_shape[2] // Win_X_shape))]

    window_num_zyx = window_num[0] * window_num[1] * window_num[2]
    window_num_yx = window_num[1] * window_num[2]
    window_num_x = window_num[2]
    window_num_y = window_num[1]
    window_num_z = window_num[0]

    # X = sparse_shape[2]
    # Y = sparse_shape[1]
    # Z = sparse_shape[0]
    # XYZ = X*Y*Z
    # XZ = X*Z
    # XY = X*Y
    #
    # index_flat = index[:,0]*XYZ+index[:,1] + index[:,2]*XZ+index[:,3]*Y

    index_Z_win = index[:, 1] // Win_Z_shape
    index_Y_win = index[:, 2] // Win_Y_shape
    index_X_win = index[:, 3] // Win_X_shape

    index_Z_in_win = index[:, 1] % Win_Z_shape

    # index_in_win = torch.stack([index[:, 1] % Win_Z_shape, index[:, 2] % Win_Y_shape, index[:, 3] % Win_X_shape]).transpose(1, 0)

    xs = Win_X_shape
    ys = Win_Y_shape
    zs = Win_Z_shape
    xzs = xs * zs
    yzs = ys * zs
    xyzs = xs * ys * zs

    depth = int(max(Win_X_shape, Win_Y_shape, Win_Z_shape) - 1).bit_length()

    index_win_X = (index[:,
                   0] * window_num_zyx + index_Z_win * window_num_yx + index_Y_win * window_num_x + index_X_win).long()  # 该voxel在第几个win

    # index_win_X_2P = (index_Z_win * window_num_yx + index_Y_win * window_num_x + index_X_win).long()

    # index_X_in_win = index[:, 1] % zs + index[:, 2] % ys * xzs + index[:, 3] % xs * zs + index_win_X * xyzs
    # _, raw2sorted_indices_index_win_X = torch.sort(index_X_in_win, dim=0)

    # code = encode(index[:, 1:], batch=index[:, 0], depth=depth, order="hilbert", num_dims=3)    # 每个voxel在H曲线中的位置
    # _, raw2sorted_indices_index_win_X = torch.sort(code + index_win_X * (2 ** depth), dim=0)    # 根据Win排序后，每个voxel在H曲线中的位置

    if serializ == 'XYZ':
        code2 = encode(index[:, 1:][:, [1, 2, 0]], batch=index[:, 0], depth=depth, order="hilbert", num_dims=3)
        code2 = code2 + index_win_X * ((2 ** depth) ** 3)
    if serializ == 'XY':
        code2 = encode(index[:, 1:][:, [1, 2]], batch=index[:, 0], depth=depth, order="hilbert", num_dims=2).long()
        code2 = code2 + index_win_X * ((2 ** depth) ** 3) + index_Z_in_win
    _, raw2sorted_indices_index_win_X = torch.sort(code2)
    # order_in_win = order+index_win_X*xyzs

    # from .show_points import show_voxel, show_voxel_win
    # show_win_voxel_coords = index[raw2sorted_indices_index_win_X]
    # show_index_win_X = index_win_X[raw2sorted_indices_index_win_X]
    # # order = torch.argsort(code)
    # # show_win_voxel_coords = index[order]
    # # show_index_win_X = index_win_X[order]
    # addcuncum = torch.ones(index.size()[0])
    # energy = torch.cumsum(addcuncum, dim=0)
    # # code, order = encode(voxel_coords[:, 1:], batch=voxel_coords[:, 0], depth=16, order="hilbert")
    # show_voxel(show_win_voxel_coords[:, [2, 3, 1]][index[:, 0].cpu()==0].float().cpu(), energy[index[:, 0].cpu()==0])  #
    # #
    # countnum_show_index_win_X = torch.bincount(show_index_win_X)
    # max_win = countnum_show_index_win_X.argmax().cpu()
    # show_voxel(show_win_voxel_coords[show_index_win_X.cpu() == max_win][:, [2, 3, 1]].float().cpu(),
    #            energy[show_index_win_X.cpu() == max_win])  #

    # raw2sorted_indices_index_win_X, sorted_countnum_voxel_in_group_in_win_X_flat, group_num_X = index_to_group(group_size, window_num, index_win_X, window_num_zyx, batch_size, device)
    sorted_countnum_voxel_in_group_in_win_X_flat, group_num_X = index_to_group(group_size, index_win_X, window_num_zyx,
                                                                               batch_size, device)

    index_win_Y = (index[:,
                   0] * window_num_zyx + index_Z_win * window_num_yx + index_X_win * window_num_y + index_Y_win).long()
    # index_Y_in_win = index[:, 1] % zs + index[:, 2] % ys * zs + index[:, 3] % xs * yzs + index_win_Y * xyzs
    # _, raw2sorted_indices_index_win_Y = torch.sort(index_Y_in_win, dim=0)

    # code = encode(index[:, 1:], batch=index[:, 0], depth=depth, order="hilbert-trans")    # 每个voxel在H曲线中的位置
    # _, raw2sorted_indices_index_win_Y = torch.sort(code+index_win_Y*(code.max()+1), dim=0)

    if serializ == 'XYZ':
        code2 = (2 ** depth) ** 3 - encode(index[:, 1:][:, [1, 2, 0]], batch=index[:, 0], depth=depth,
                                           order="hilbert-trans", num_dims=3)
        # code2 = code2 + index_win_Y * ((2 ** depth) ** 3)
        code2 = code2 + index_win_X * ((2 ** depth) ** 3)
    if serializ == 'XY':
        code2 = (2 ** depth) ** 3 - encode(index[:, 1:][:, [1, 2]], batch=index[:, 0], depth=depth,
                                           order="hilbert-trans", num_dims=2).long()
        # code2 = code2 + index_win_Y * ((2 ** depth) ** 3) + index_Z_in_win
        code2 = code2 + index_win_X * ((2 ** depth) ** 3) + index_Z_in_win

    _, raw2sorted_indices_index_win_Y = torch.sort(code2)

    # from .show_points import show_voxel, show_voxel_win
    # show_win_voxel_coords = index[raw2sorted_indices_index_win_Y]
    # show_index_win_Y = index_win_Y[raw2sorted_indices_index_win_Y]
    # addcuncum = torch.ones(index.size()[0])
    # energy = torch.cumsum(addcuncum, dim=0)  # voxel在第几个group第几个值
    # # code, order = encode(voxel_coords[:, 1:], batch=voxel_coords[:, 0], depth=16, order="hilbert")
    # show_voxel(show_win_voxel_coords[:, [2, 3, 1]].float().cpu()[:], energy[:])  #
    #
    # # countnum_show_index_win_Y = torch.bincount(show_index_win_Y)
    # # max_win = countnum_show_index_win_Y.argmax().cpu()-1
    # # show_voxel(show_win_voxel_coords[show_index_win_Y.cpu() == max_win][:, [2, 3, 1]].float().cpu(), energy[show_index_win_Y.cpu() == max_win]) #

    # raw2sorted_indices_index_win_Y, sorted_countnum_voxel_in_group_in_win_Y_flat, group_num_Y = index_to_group(group_size, window_num, index_win_Y, window_num_zyx, batch_size, device, index_Y_in_win)
    # sorted_countnum_voxel_in_group_in_win_Y_flat, group_num_Y = index_to_group(group_size, index_win_Y, window_num_zyx, batch_size, device)

    # sorted_countnum_voxel_in_group_in_win_Y_flat, group_num_Y = index_to_group(group_size, index_win_X, window_num_zyx,
    #                                                                            batch_size, device)

    sorted_countnum_voxel_in_group_in_win_Y_flat = sorted_countnum_voxel_in_group_in_win_X_flat
    group_num_Y = group_num_X

    mappings = {'raw2sorted_indices_index_win_X': raw2sorted_indices_index_win_X,
                'sorted_countnum_voxel_in_group_in_win_X_flat': sorted_countnum_voxel_in_group_in_win_X_flat,
                'raw2sorted_indices_index_win_Y': raw2sorted_indices_index_win_Y,
                'sorted_countnum_voxel_in_group_in_win_Y_flat': sorted_countnum_voxel_in_group_in_win_Y_flat,
                'group_num_X': group_num_X,
                'group_num_Y': group_num_Y
                }

    return mappings


class win_and_group_tf(nn.Module):
    def __init__(
            self,
            dim,
            nhead,
            window_num,
            group_size,
            operator_cfg,
            n_layer,
            cross=True,
            posemb=True

    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.window_num = window_num  # nucs,6米
        self.h = nhead
        self.cross = cross
        self.posemb = posemb
        # self.self_att1 = self_att_Layer(dim, group_size, nhead, dim_feedforward=2048, dropout=0.1)    #
        # self.self_att2 = self_att_Layer(dim, group_size, nhead, dim_feedforward=2048, dropout=0.1)    #
        self.self_att1 = cross_att_Layer(dim, group_size, nhead, dim_feedforward=2048, dropout=0.1)  #
        self.self_att2 = cross_att_Layer(dim, group_size, nhead, dim_feedforward=2048, dropout=0.1)  #
        self.cross_att1 = cross_att_Layer(dim, group_size, nhead, dim_feedforward=2048, dropout=0.1)  #
        self.cross_att2 = cross_att_Layer(dim, group_size, nhead, dim_feedforward=2048, dropout=0.1)  #

        if posemb:
            self.LBP_posembed = LBPEmbeddingLearned(3, dim, 40)
            # self.self_posembed = PositionEmbeddingLearned(3, dim)

        self.sub_conv = spconv.SparseSequential(
            spconv.SubMConv3d(dim, dim, 3, bias=False, indice_key='subm'),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def point2group_X(self, x, mappings, device, dim):
        raw2win_X = x[mappings['raw2sorted_indices_index_win_X']]
        x_g_X = torch.zeros(mappings['group_num_X'] * self.group_size, dim).to(x.dtype).to(device)
        x_g_X[mappings['sorted_countnum_voxel_in_group_in_win_X_flat']] = raw2win_X
        x_g_X = x_g_X.view(mappings['group_num_X'], self.group_size, dim)
        return x_g_X

    def point2group_Y(self, x, mappings, device, dim):
        raw2win_Y = x[mappings['raw2sorted_indices_index_win_Y']]
        x_g_Y = torch.zeros(mappings['group_num_Y'] * self.group_size, dim).to(x.dtype).to(device)
        x_g_Y[mappings['sorted_countnum_voxel_in_group_in_win_Y_flat']] = raw2win_Y
        x_g_Y = x_g_Y.view(mappings['group_num_Y'], self.group_size, dim)
        return x_g_Y

    def group_mask_X(self, mappings, device):
        mask_X = torch.zeros((mappings['group_num_X'] * self.group_size, self.h, 1), dtype=torch.bool).to(device)
        mask_X[mappings['sorted_countnum_voxel_in_group_in_win_X_flat']] = True
        mask_X = mask_X.view(mappings['group_num_X'], self.group_size, self.h, 1).transpose(2, 1)
        return mask_X

    def group_mask_Y(self, mappings, device):
        mask_Y = torch.zeros((mappings['group_num_Y'] * self.group_size, self.h, 1), dtype=torch.bool).to(device)
        mask_Y[mappings['sorted_countnum_voxel_in_group_in_win_Y_flat']] = True
        mask_Y = mask_Y.view(mappings['group_num_Y'], self.group_size, self.h, 1).transpose(2, 1)
        return mask_Y

    def forward(self, x, mappings):
        x_features = x.features.clone()

        index = x.indices
        batch_size = x.batch_size
        sparse_shape = x.spatial_shape
        dim = x.features.size()[1]
        device = x.indices.device

        if self.posemb:
            x_pos_embed = self.LBP_posembed(x)
            # x_pos_embed = self.self_posembed(x.indices.float())
            x_features = x_features + x_pos_embed

        # mappings = sprace_to_groupfearutes(self.group_size, self.window_num, index = index, batch_size = batch_size, sparse_shape = sparse_shape, dim = dim,  device = device)

        x_features_g_X = self.point2group_X(x_features, mappings, device, dim)
        # x_index_g_X = self.point2group_X(index, mappings, device, 4)
        mask_X = self.group_mask_X(mappings, device)
        x_features_g_X1 = self.self_att1(query_features=x_features_g_X, KV_features=x_features_g_X, mask=mask_X)
        # x_features_g_X = self.cross_att1(x_features_g_X1, x_index_g_X, x_features_g_X, mask_X)
        # x_features[mappings['raw2sorted_indices_index_win_X']] = x_features_g_X.view(mappings['group_num_X']* self.group_size, dim)[mappings['sorted_countnum_voxel_in_group_in_win_X_flat']]

        x_features_g_Y = self.point2group_Y(x_features, mappings, device, dim)
        # x_index_g_Y = self.point2group_Y(index, mappings, device, 4)
        mask_Y = self.group_mask_Y(mappings, device)
        x_features_g_Y1 = self.self_att2(query_features=x_features_g_Y, KV_features=x_features_g_Y, mask=mask_Y)
        # x_features_g_Y = self.cross_att2(x_features_g_Y1, x_index_g_Y, x_features_g_Y, mask_Y)
        # x_features[mappings['raw2sorted_indices_index_win_Y']] = x_features_g_Y.view(mappings['group_num_Y']*self.group_size, dim)[mappings['sorted_countnum_voxel_in_group_in_win_Y_flat']]

        if self.cross:
            # x_features_Y = x_features.clone()
            # x_features_Y[mappings['raw2sorted_indices_index_win_Y']] = \
            #     x_features_g_Y1.view(mappings['group_num_Y'] * self.group_size, dim)[
            #         mappings['sorted_countnum_voxel_in_group_in_win_Y_flat']]
            # x_features_g_Y2X = self.point2group_X(x_features_Y, mappings, device, dim)

            # x_features_X = x_features.clone()
            # x_features_X[mappings['raw2sorted_indices_index_win_X']] = \
            #     x_features_g_X1.view(mappings['group_num_X'] * self.group_size, dim)[
            #         mappings['sorted_countnum_voxel_in_group_in_win_X_flat']]
            # x_features_g_X2Y = self.point2group_Y(x_features_X, mappings, device, dim)

            # x_features_g_X = self.cross_att1(query_features=x_features_g_Y2X, KV_features=x_features_g_X1, mask=mask_X)
            # x_features_g_Y = self.cross_att2(query_features=x_features_g_X2Y, KV_features=x_features_g_Y1, mask=mask_Y)

            x_features_g_X = self.cross_att1(query_features=x_features_g_Y1, KV_features=x_features_g_X1, mask=mask_X)
            x_features_g_Y = self.cross_att2(query_features=x_features_g_X1, KV_features=x_features_g_Y1, mask=mask_Y)
        else:
            x_features_g_X = self.cross_att1(query_features=x_features_g_X1, KV_features=x_features_g_X1, mask=mask_X)
            x_features_g_Y = self.cross_att2(query_features=x_features_g_Y1, KV_features=x_features_g_Y1, mask=mask_Y)

        x_features_TF = x_features.clone()
        x_features_TF[mappings['raw2sorted_indices_index_win_X']] = \
        x_features_g_X.view(mappings['group_num_X'] * self.group_size, dim)[
            mappings['sorted_countnum_voxel_in_group_in_win_X_flat']]
        x_features_TF[mappings['raw2sorted_indices_index_win_Y']] += \
        x_features_g_Y.view(mappings['group_num_Y'] * self.group_size, dim)[
            mappings['sorted_countnum_voxel_in_group_in_win_Y_flat']]

        x_tf = spconv.SparseConvTensor(
            features=x_features_TF.contiguous(),
            indices=x.indices,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )
        x_tf = self.sub_conv(x_tf)
        x = replace_feature(x, x_tf.features)  # +x.features

        # x_features[mappings['raw2sorted_indices_index_win_X']] += \
        #     x_features_g_X.view(mappings['group_num_X'] * self.group_size, dim)[
        #         mappings['sorted_countnum_voxel_in_group_in_win_X_flat']]
        # x_features[mappings['raw2sorted_indices_index_win_Y']] += \
        #     x_features_g_Y.view(mappings['group_num_Y'] * self.group_size, dim)[
        #         mappings['sorted_countnum_voxel_in_group_in_win_Y_flat']]
        # x = replace_feature(x, x_features)

        return x


