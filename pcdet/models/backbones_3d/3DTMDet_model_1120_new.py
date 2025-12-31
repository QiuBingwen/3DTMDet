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

# from .grouptf_utils import win_and_group_tf, sprace_to_groupfearutes, win_and_group_selfatt, win_and_group_crossatt
from .grouptf_serializ_utils import win_and_group_tf, sprace_to_groupfearutes
# from .grouptf_serializ_utils import encode

# from .show_points import show_voxel, show_voxel_win

knnquery = KNNQuery.apply

@torch.inference_mode()
def get_window_coors_shift_v2(coords, sparse_shape, window_shape, shift=False):
    sparse_shape_z, sparse_shape_y, sparse_shape_x = sparse_shape
    win_shape_x, win_shape_y, win_shape_z = window_shape

    if shift:
        shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    else:
        shift_x, shift_y, shift_z = 0, 0, 0  # win_shape_x, win_shape_y, win_shape_z

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1)  # plus one here to meet the needs of shift.

    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    x = coords[:, 3] + shift_x
    y = coords[:, 2] + shift_y
    z = coords[:, 1] + shift_z

    win_coors_x = x // win_shape_x
    win_coors_y = y // win_shape_y
    win_coors_z = z // win_shape_z

    coors_in_win_x = x % win_shape_x
    coors_in_win_y = y % win_shape_y
    coors_in_win_z = z % win_shape_z

    batch_win_inds_x = coords[:, 0] * max_num_win_per_sample + win_coors_x * max_num_win_y * max_num_win_z + \
                       win_coors_y * max_num_win_z + win_coors_z
    batch_win_inds_y = coords[:, 0] * max_num_win_per_sample + win_coors_y * max_num_win_x * max_num_win_z + \
                       win_coors_x * max_num_win_z + win_coors_z

    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)

    return batch_win_inds_x, batch_win_inds_y, coors_in_win


def get_window_coors_shift_v1(coords, sparse_shape, window_shape):
    _, m, n = sparse_shape
    n2, m2, _ = window_shape

    n1 = int(np.ceil(n / n2) + 1)  # plus one here to meet the needs of shift.
    m1 = int(np.ceil(m / m2) + 1)  # plus one here to meet the needs of shift.

    x = coords[:, 3]
    y = coords[:, 2]

    x1 = x // n2
    y1 = y // m2
    x2 = x % n2
    y2 = y % m2

    return 2 * n2, 2 * m2, 2 * n1, 2 * m1, x1, y1, x2, y2


class FlattenedWindowMapping(nn.Module):
    def __init__(
            self,
            window_shape,
            group_size,
            shift,
            win_version='v2'
    ) -> None:
        super().__init__()
        self.window_shape = window_shape
        self.group_size = group_size
        self.win_version = win_version
        self.shift = shift

    def forward(self, coords: torch.Tensor, batch_size: int, sparse_shape: list):
        coords = coords.long()
        _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True)
        batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
        num_per_batch_p = (
                torch.div(
                    batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1,
                    self.group_size,
                    rounding_mode="trunc",
                )
                * self.group_size
        )

        batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))
        flat2win = torch.arange(batch_start_indices_p[-1], device=coords.device)  # .to(coords.device)
        win2flat = torch.arange(batch_start_indices[-1], device=coords.device)  # .to(coords.device)

        for i in range(batch_size):
            if num_per_batch[i] != num_per_batch_p[i]:
                bias_index = batch_start_indices_p[i] - batch_start_indices[i]

                # print(batch_start_indices_p[i + 1] + (num_per_batch[i] % self.group_size) - self.group_size == batch_start_indices_p[i]+num_per_batch[i])

                flat2win[batch_start_indices_p[i] + num_per_batch[i]:batch_start_indices_p[i + 1]] \
                    = flat2win[batch_start_indices_p[i] + num_per_batch[i] - self.group_size: batch_start_indices_p[i + 1] - self.group_size] \
                    if (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) - self.group_size != 0 else \
                    win2flat[batch_start_indices[i]: batch_start_indices[i + 1]].repeat(
                        (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) // num_per_batch[i] + 1)[
                    : self.group_size - (num_per_batch[i] % self.group_size)] + bias_index

            win2flat[batch_start_indices[i]: batch_start_indices[i + 1]] += (
                    batch_start_indices_p[i] - batch_start_indices[i]
            )

            flat2win[batch_start_indices_p[i]: batch_start_indices_p[i + 1]] -= (
                    batch_start_indices_p[i] - batch_start_indices[i]
            )

        mappings = {"flat2win": flat2win, "win2flat": win2flat}

        get_win = self.win_version

        if get_win == 'v1':
            for shifted in [False]:
                (
                    n2,
                    m2,
                    n1,
                    m1,
                    x1,
                    y1,
                    x2,
                    y2,
                ) = get_window_coors_shift_v1(coords, sparse_shape, self.window_shape)
                vx = (n1 * y1 + (-1) ** y1 * x1) * n2 * m2 + (-1) ** y1 * (m2 * x2 + (-1) ** x2 * y2)
                vx += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
                vy = (m1 * x1 + (-1) ** x1 * y1) * m2 * n2 + (-1) ** x1 * (n2 * y2 + (-1) ** y2 * x2)
                vy += coords[:, 0] * sparse_shape[2] * sparse_shape[1] * sparse_shape[0]
                _, mappings["x" + ("_shift" if shifted else "")] = torch.sort(vx)
                _, mappings["y" + ("_shift" if shifted else "")] = torch.sort(vy)

        elif get_win == 'v2':
            batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
                                                                                         self.window_shape, self.shift)
            vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vx += coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                  self.window_shape[2] + coors_in_win[..., 0]

            vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vy += coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                  self.window_shape[2] + coors_in_win[..., 0]

            _, mappings["x"] = torch.sort(vx)
            _, mappings["y"] = torch.sort(vy)

        elif get_win == 'v3':
            batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
                                                                                         self.window_shape)
            vx = batch_win_inds_x * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vx_xy = vx + coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                    self.window_shape[2] + coors_in_win[..., 0]
            vx_yx = vx + coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                    self.window_shape[2] + coors_in_win[..., 0]

            vy = batch_win_inds_y * self.window_shape[0] * self.window_shape[1] * self.window_shape[2]
            vy_xy = vy + coors_in_win[..., 2] * self.window_shape[1] * self.window_shape[2] + coors_in_win[..., 1] * \
                    self.window_shape[2] + coors_in_win[..., 0]
            vy_yx = vy + coors_in_win[..., 1] * self.window_shape[0] * self.window_shape[2] + coors_in_win[..., 2] * \
                    self.window_shape[2] + coors_in_win[..., 0]

            _, mappings["x_xy"] = torch.sort(vx_xy)
            _, mappings["y_xy"] = torch.sort(vy_xy)
            _, mappings["x_yx"] = torch.sort(vx_yx)
            _, mappings["y_yx"] = torch.sort(vy_yx)

        return mappings


class PatchMerging3D(nn.Module):
    def __init__(self, dim, out_dim=-1, down_scale=[2, 2, 2], norm_layer=nn.LayerNorm, diffusion=False, diff_scale=0.2):
        super().__init__()
        self.dim = dim

        self.sub_conv = spconv.SparseSequential(
            spconv.SubMConv3d(dim, dim, 3, bias=False, indice_key='subm'),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        # if out_dim == -1:
        #     self.norm = norm_layer(dim)
        # else:
        #     self.norm = norm_layer(out_dim)

        self.sigmoid = nn.Sigmoid()
        self.down_scale = down_scale
        self.diffusion = diffusion
        self.diff_scale = diff_scale

        self.num_points = 6 #3

        # if self.diffusion:
        #     self.attn_layers = TF_att_Layer(dim, 8)

    def forward(self, x, coords_shift=1, diffusion_scale=3):
        assert diffusion_scale==4 or diffusion_scale==2 or diffusion_scale==3
        x = self.sub_conv(x)

        d, h, w = x.spatial_shape
        down_scale = self.down_scale

        if self.diffusion:

            # src2, x_feat_att = self.attn_layers(x)
            x_feat_att = x.features.mean(-1)
            batch_size = x.indices[:, 0].max() + 1
            selected_diffusion_feats_list = [x.features.clone()] #  + src2
            selected_diffusion_coords_list = [x.indices.clone()]
            for i in range(batch_size):
                mask = x.indices[:, 0] == i
                valid_num = mask.sum()
                K = int(valid_num * self.diff_scale)
                _, indices = torch.topk(x_feat_att[mask], K)

                selected_coords_copy = x.indices[mask][indices].clone()
                selected_coords_num = selected_coords_copy.shape[0]
                selected_coords_expand = selected_coords_copy.repeat(diffusion_scale, 1)
                selected_feats_expand = x.features[mask][indices].repeat(diffusion_scale, 1) * 0.0

                if diffusion_scale==3:
                    spatial_shape = np.array(x.spatial_shape)
                    spatial_shape_middle = torch.tensor(spatial_shape/2 + 0.5).to(x.indices.device)
                    diffusion_direction = ((selected_coords_copy[:, 1:] - spatial_shape_middle) / abs(selected_coords_copy[:, 1:] - spatial_shape_middle)).int()

                    selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 3:4] = (selected_coords_copy[:, 3:4] + diffusion_direction[:, 2:3]*coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 2:3] = (selected_coords_copy[:, 2:3]).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 1:2] = (selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                    selected_coords_expand[selected_coords_num * 1:selected_coords_num * 2, 3:4] = (selected_coords_copy[:, 3:4]).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 1:selected_coords_num * 2, 2:3] = (selected_coords_copy[:, 2:3] + diffusion_direction[:, 1:2]*coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 1:selected_coords_num * 2, 1:2] = (selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 3:4] = (selected_coords_copy[:, 3:4] + diffusion_direction[:, 2:3]*coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 2:3] = (selected_coords_copy[:, 2:3] + diffusion_direction[:, 1:2]*coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 1:2] = (selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                if diffusion_scale == 4 or diffusion_scale==2:
                    selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 3:4] = (selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 2:3] = (selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 0:selected_coords_num * 1, 1:2] = (selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                    selected_coords_expand[selected_coords_num:selected_coords_num * 2, 3:4] = (selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num:selected_coords_num * 2, 2:3] = (selected_coords_copy[:, 2:3] + coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num:selected_coords_num * 2, 1:2] = (selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                if diffusion_scale==4:
#                         print('####diffusion_scale==4')
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 3:4] = (selected_coords_copy[:, 3:4] - coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 2:3] = (selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 1:2] = (selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 3:4] = (selected_coords_copy[:, 3:4] + coords_shift).clamp(min=0, max=w - 1)
                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 2:3] = (selected_coords_copy[:, 2:3] - coords_shift).clamp(min=0, max=h - 1)
                    selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 1:2] = (selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

                selected_diffusion_coords_list.append(selected_coords_expand)
                selected_diffusion_feats_list.append(selected_feats_expand)

            coords = torch.cat(selected_diffusion_coords_list)
            final_diffusion_feats = torch.cat(selected_diffusion_feats_list)

        else:
            coords = x.indices.clone()
            final_diffusion_feats = x.features.clone()

        coords[:, 3:4] = coords[:, 3:4] // down_scale[0]
        coords[:, 2:3] = coords[:, 2:3] // down_scale[1]
        coords[:, 1:2] = coords[:, 1:2] // down_scale[2]

        scale_xyz = (-(-x.spatial_shape[0] // down_scale[2])) * (x.spatial_shape[1] // down_scale[1]) * (
                x.spatial_shape[2] // down_scale[0])
        scale_yz = (-(-x.spatial_shape[0] // down_scale[2])) * (x.spatial_shape[1] // down_scale[1])
        scale_z = (-(-x.spatial_shape[0] // down_scale[2]))


        merge_coords = coords[:, 0].int() * scale_xyz + coords[:, 3] * scale_yz + coords[:, 2] * scale_z + coords[:, 1]

        features_expand = final_diffusion_feats

        new_sparse_shape = [math.ceil(x.spatial_shape[i] / down_scale[2 - i]) for i in range(3)]
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True, return_counts=False, dim=0)

        x_merge = torch_scatter.scatter_add(features_expand, unq_inv, dim=0)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // scale_xyz,
                                    (unq_coords % scale_xyz) // scale_yz,
                                    (unq_coords % scale_yz) // scale_z,
                                    unq_coords % scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        # x_merge = self.norm(x_merge).contiguous()

        x_merge = spconv.SparseConvTensor(
            features=x_merge.contiguous(),
            indices=voxel_coords.int(),
            spatial_shape=new_sparse_shape,
            batch_size=x.batch_size
        )
        return x_merge, unq_inv


class PatchExpanding3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, up_x, unq_inv, link = True):
        # z, y, x
        n, c = x.features.shape

        x_copy = torch.gather(x.features, 0, unq_inv.unsqueeze(1).repeat(1, c)).contiguous()
        if link:
            up_x = up_x.replace_feature(up_x.features + x_copy)
        else:
            up_x = up_x.replace_feature(x_copy)

        return up_x


class LocalGlobalBlock(nn.Module):
    def __init__(self, dim, group_size):
        super().__init__()
        self.group_size = group_size
        # self.MLP = nn.Linear(in_features=128, out_features=64).cuda()
        # self.LN = nn.LayerNorm(64).cuda()

        self.MLP = nn.Sequential(nn.Linear(in_features=dim*2, out_features=dim),
                                 nn.LayerNorm(dim),
                                 nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x_features):
        global_features = torch.max(x_features, dim=-2, keepdim=False)[0]
        global_features = global_features.view(x_features.shape[0], 1, x_features.shape[-1]).repeat(1, self.group_size, 1)
        x_features = torch.cat((x_features, global_features), dim=-1)

        x_features = self.MLP(x_features)
        # x_features = self.LN(x_features)

        return x_features


class EB_LG_Block(nn.Module):
    def __init__(self, dim, group_size):
        super().__init__()
        self.group_size = group_size
        self.LG_block = nn.Sequential(LocalGlobalBlock(dim, self.group_size))
        self.MLP = nn.Sequential(nn.Linear(in_features=dim, out_features=dim),
                                 nn.LayerNorm(dim),
                                 nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x_features):
        hidden_features = self.LG_block(x_features)
        error_features = self.MLP(hidden_features - x_features)
        x_features = x_features + error_features

        # x_features = self.LN(x_features)

        return x_features


LinearOperatorMap = {
    'Mamba': MambaBlock,
    'RWKV': RWKVBlock,
    'RetNet': RetNetBlock,
    'xLSTM': xLSTM_Block,
    'TTT': TTTBlock,
}



class LBPEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288, nsample = 16):
        super().__init__()
        self.nsample = nsample
        self.position_embedding_head = nn.Sequential(
            nn.Linear((self.nsample-1)*input_channel, num_pos_feats),
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
        x_indices_n = x.indices[:,1:][[idx]]
        x_indices_pos  = (x_indices_n[:, 1:, :] - x_indices_n[:, 0:1, :]).reshape(-1, (self.nsample-1)*3 )
        position_embedding = self.position_embedding_head(x_indices_pos.float())
        return position_embedding



class LIONLayer(nn.Module):
    def __init__(self, dim, nums, window_shape, group_size, direction, shift, operator=None, layer_id=0, n_layer=0):
        super(LIONLayer, self).__init__()

        self.window_shape = window_shape
        self.group_size = group_size
        self.dim = dim
        self.direction = direction

        operator_cfg = operator.CFG
        operator_cfg['d_model'] = dim

        block_list = []
        for i in range(len(direction)):
            operator_cfg['layer_id'] = i + layer_id
            operator_cfg['n_layer'] = n_layer
            # operator_cfg['with_cp'] = layer_id >= 16
            operator_cfg['with_cp'] = layer_id >= 0 ## all lion layer use checkpoint to save GPU memory!! (less 24G for training all models!!!)
            print('### use part of checkpoint!!')
            block_list.append(LinearOperatorMap[operator.NAME](**operator_cfg))

        self.blocks = nn.ModuleList(block_list)
        self.window_partition = FlattenedWindowMapping(self.window_shape, self.group_size, shift)

    def forward(self, x):
        mappings = self.window_partition(x.indices, x.batch_size, x.spatial_shape)

        for i, block in enumerate(self.blocks):
            indices = mappings[self.direction[i]]
            x_features = x.features[indices][mappings["flat2win"]]
            x_features = x_features.view(-1, self.group_size, x.features.shape[-1])

            x_features = block(x_features)

            x.features[indices] = x_features.view(-1, x_features.shape[-1])[mappings["win2flat"]]

        return x



class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats))

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class TFLayer_out(nn.Module):
    def __init__(self, dim: int, group_size_tf, operator=None, n_layer=0, window_nums=[1, 12, 12], if_mamba = True, if_Middle_layer = True):
        super().__init__()

        self.dim = dim
        # window_nums = [1, 12, 12]
        group_size = 16
        self.group_size = group_size
        self.window_nums = window_nums
        self.Middle_layer = win_and_group_tf(dim, 8, window_nums, group_size, operator, n_layer)
        # self.Middle_layer = LIONLayer(dim, 1, [13, 13, 4], group_size_mamba/4, direction, True, operator, layer_id, n_layer)
        self.Middle_pos_emb = PositionEmbeddingLearned(input_channel=3, num_pos_feats=dim)


    def forward(self, x):
        device = x.features.device
        mapping = sprace_to_groupfearutes(self.group_size, self.window_nums, index=x.indices,
                                          batch_size=x.batch_size,
                                          sparse_shape=x.spatial_shape, dim=self.dim, device=device)
        pos_emb = self.get_pos_embed(spatial_shape=x.spatial_shape, coors=x.indices[:, 1:],
                                     embed_layer=self.Middle_pos_emb)
        x = replace_feature(x, pos_emb + x.features)  # x + pos_emb
        x = self.Middle_layer(x, mapping)
        return x


    def get_pos_embed(self, spatial_shape, coors, embed_layer, normalize_pos=True):
        window_shape = spatial_shape[::-1]  # spatial_shape:   win_z, win_y, win_x ---> win_x, win_y, win_z
        embed_layer = embed_layer
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3
        z, y, x = coors[:, 0] - win_z / 2, coors[:, 1] - win_y / 2, coors[:, 2] - win_x / 2
        if normalize_pos:
            if ndim == 2:
                x = x / win_x * 2 * 3.1415  # [-pi, pi]
                y = y / win_y * 2 * 3.1415  # [-pi, pi]
            else:
                x = x / win_x * 2 * 3.1415  # [-pi, pi]
                y = y / win_y * 2 * 3.1415  # [-pi, pi]
                z = z / win_z * 2 * 3.1415  # [-pi, pi]
        if ndim == 2:
            location = torch.stack((x, y, z), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location.contiguous())
        return pos_embed

class PatchDiffusion3D(nn.Module):
    def __init__(self, dim, diff_scale=0.25, conv=True):
        super().__init__()

        self.diffusion = True
        self.diff_scale = diff_scale

        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        # if self.diffusion:
        #     self.attn_layers = TF_att_Layer(dim, 8)

        self.conv = conv
        if conv:
            self.sub_conv = spconv.SparseSequential(
                spconv.SubMConv3d(dim, dim, 3, bias=False, indice_key='subm'),
                nn.LayerNorm(dim),
                nn.GELU(),
            )

        self.sub_conv2 = spconv.SparseSequential(
            spconv.SubMConv3d(dim, dim, 3, bias=False, indice_key='subm'),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(self, x, coords_shift=1, diffusion_scale=5):
        assert diffusion_scale == 4 or diffusion_scale == 2 or diffusion_scale == 3 or diffusion_scale == 5

        if self.conv:
            # x = self.sub_conv(x)
            x = self.sub_conv(x)

        d, h, w = x.spatial_shape

        valid_num = x.indices.shape[0]
        K = int(valid_num * self.diff_scale)

        # src2, x_feat_att = self.attn_layers(x)
        # x_feat_att = x.features.mean(-1)
        batch_size = x.indices[:, 0].max() + 1
        selected_diffusion_feats_list = []  # + src2
        selected_diffusion_coords_list = []

        xfeatures = x.features.clone()

        for i in range(batch_size):
            mask = x.indices[:, 0] == i

            selected_coords_copy = x.indices[mask].clone()
            selected_coords_num = selected_coords_copy.shape[0]
            selected_coords_expand = selected_coords_copy.repeat(diffusion_scale + 1, 1)
            selected_feats_expand = xfeatures[mask].repeat(diffusion_scale + 1, 1)

            spatial_shape = np.array(x.spatial_shape)
            spatial_shape_middle = torch.tensor(spatial_shape / 2 + 0.5).to(x.indices.device)
            diffusion_direction = ((selected_coords_copy[:, 1:] - spatial_shape_middle) / abs(
                selected_coords_copy[:, 1:] - spatial_shape_middle)).int()

            selected_coords_expand[selected_coords_num * 1:selected_coords_num * 2, 3:4] = (selected_coords_copy[:, 3:4] + diffusion_direction[:, 2:3] * coords_shift).clamp(min=0,max=w - 1)
            selected_coords_expand[selected_coords_num * 1:selected_coords_num * 2, 2:3] = (selected_coords_copy[:, 2:3]).clamp(min=0, max=h - 1)
            selected_coords_expand[selected_coords_num * 1:selected_coords_num * 2, 1:2] = (selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

            selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 3:4] = (selected_coords_copy[:, 3:4]).clamp(min=0, max=w - 1)
            selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 2:3] = (selected_coords_copy[:, 2:3] + diffusion_direction[:, 1:2] * coords_shift).clamp(min=0,max=h - 1)
            selected_coords_expand[selected_coords_num * 2:selected_coords_num * 3, 1:2] = (selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

            selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 3:4] = (selected_coords_copy[:, 3:4] + diffusion_direction[:, 2:3] * coords_shift).clamp(min=0,max=w - 1)
            selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 2:3] = (selected_coords_copy[:, 2:3] + diffusion_direction[:, 1:2] * coords_shift).clamp(min=0,max=h - 1)
            selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 1:2] = (selected_coords_copy[:, 1:2]).clamp(min=0, max=d - 1)

            if diffusion_scale==5:
                selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 3:4] = (selected_coords_copy[:, 3:4]).clamp(min=0, max=w - 1)
                selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 2:3] = (selected_coords_copy[:, 2:3]).clamp(min=0, max=h - 1)
                selected_coords_expand[selected_coords_num * 3:selected_coords_num * 4, 1:2] = (selected_coords_copy[:, 1:2]-1).clamp(min=0, max=d - 1)

                selected_coords_expand[selected_coords_num * 4:selected_coords_num * 5, 3:4] = (selected_coords_copy[:, 3:4]).clamp(min=0, max=w - 1)
                selected_coords_expand[selected_coords_num * 4:selected_coords_num * 5, 2:3] = (selected_coords_copy[:, 2:3]).clamp(min=0, max=h - 1)
                selected_coords_expand[selected_coords_num * 4:selected_coords_num * 5, 1:2] = (selected_coords_copy[:, 1:2]+1).clamp(min=0, max=d - 1)

            selected_diffusion_coords_list.append(selected_coords_expand)
            selected_diffusion_feats_list.append(selected_feats_expand)

        coords = torch.cat(selected_diffusion_coords_list)
        features_diff = torch.cat(selected_diffusion_feats_list)

        ############### x p ###############
        scale_xyz = x.spatial_shape[0] * x.spatial_shape[1] * x.spatial_shape[2]
        scale_yz = x.spatial_shape[0] * x.spatial_shape[1]
        scale_z = x.spatial_shape[0]
        sparse_shape = [math.ceil(x.spatial_shape[i]) for i in range(3)]

        ############### diffusion_sellect ###############
        # selected_coords_expand[:, 3:4] = selected_coords_expand[:, 3:4]
        # selected_coords_expand[:, 2:3] = selected_coords_expand[:, 2:3]
        # selected_coords_expand[:, 1:2] = selected_coords_expand[:, 1:2]
        coords_diff = coords[:, 0].int() * scale_xyz + coords[:, 3] * scale_yz + coords[:, 2] * scale_z + coords[:, 1]

        unq_coords_diff, unq_inv_diff, norm_grouped_num = torch.unique(coords_diff, return_inverse=True,
                                                                       return_counts=True, dim=0)
        norm_grouped_num = norm_grouped_num.unsqueeze(-1)

        x_diff_tmp = torch_scatter.scatter_add(features_diff, unq_inv_diff, dim=0)
        x_diff_tmp = x_diff_tmp / norm_grouped_num

        unq_coords_diff = unq_coords_diff.int()
        voxel_coords = torch.stack((unq_coords_diff // scale_xyz,
                                    (unq_coords_diff % scale_xyz) // scale_yz,
                                    (unq_coords_diff % scale_yz) // scale_z,
                                    unq_coords_diff % scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]].contiguous()
        x_merge_diff_tmp = self.norm(x_diff_tmp).contiguous()
        x_merge_diff_tmp = spconv.SparseConvTensor(
            features=x_merge_diff_tmp.contiguous(),
            indices=voxel_coords.int(),
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )

        x_merge_diff_tmp = self.sub_conv2(x_merge_diff_tmp)

        x_diff_features_tmp = x_merge_diff_tmp.features
        x_diff_indices_tmp = x_merge_diff_tmp.indices

        x_indices_raw = x.indices.clone()

        coords_diff_tmp = x_diff_indices_tmp[:, 0].int() * scale_xyz + x_diff_indices_tmp[:,
                                                                       3] * scale_yz + x_diff_indices_tmp[:,
                                                                                       2] * scale_z + x_diff_indices_tmp[
                                                                                                      :, 1]
        coords_raw = x_indices_raw[:, 0].int() * scale_xyz + x_indices_raw[:, 3] * scale_yz + x_indices_raw[:,
                                                                                              2] * scale_z + x_indices_raw[
                                                                                                             :, 1]

        tensor_is_not_in = torch.isin(coords_diff_tmp, coords_raw, invert=True)

        x_raw_features = x_diff_features_tmp[~tensor_is_not_in]
        x_raw_indices = x_diff_indices_tmp[~tensor_is_not_in]

        x_diff_features = x_diff_features_tmp[tensor_is_not_in]
        x_diff_indices = x_diff_indices_tmp[tensor_is_not_in]

        x_diff_indices_list = [x_raw_indices]
        x_diff_features_list = [x_raw_features]

        x_diff_feat_act = x_diff_features.mean(-1)
        for i in range(batch_size):
            mask = x_diff_indices[:, 0] == i
            valid_num = mask.sum()
            K = int(valid_num * self.diff_scale)
            _, indices_diff_selected = torch.topk(x_diff_feat_act[mask], K)

            x_diff_indices_list.append(x_diff_indices[mask][indices_diff_selected])
            x_diff_features_list.append(x_diff_features[mask][indices_diff_selected])

        coords = torch.cat(x_diff_indices_list)

        final_diffusion_feats = torch.cat(x_diff_features_list)

        # selected_diffusion_feats_list = []  # + src2
        # selected_diffusion_coords_list = []
        # for i in range(batch_size):
        #     mask = coords[:, 0] == i
        #     selected_diffusion_feats_list.append(final_diffusion_feats[mask])
        #     selected_diffusion_coords_list.append(coords[mask])
        #
        # coords = torch.cat([x_diff_selected_indices, x_raw_indices]).contiguous()
        # final_diffusion_feats = torch.cat(selected_diffusion_feats_list).contiguous()

        x_merge = spconv.SparseConvTensor(
            features=final_diffusion_feats.contiguous(),
            indices=coords.int(),
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )

        return x_merge





class GPTFBlock(nn.Module):
    def __init__(self, dim: int, depth: int, down_scales: list, window_shape, group_size_mamba, group_size_tf, direction, shift=False,
                 operator=None, layer_id=0, n_layer=0, window_nums_tf=[[1, 18, 18],[1, 6, 6]], if_mamba = True, if_Middle_layer = True, serializ='XYZ', diffusion=True):
        super().__init__()

        self.dim = dim

        self.serializ = serializ

        self.down_scales = down_scales
        # self.down_scales = [[2,2,2],[2,2,2],[2,2,2]]

        self.encoder = nn.ModuleList()
        self.downsample_list = nn.ModuleList()
        self.pos_emb_list = nn.ModuleList()

        norm_fn = partial(nn.LayerNorm)

        # depth = 2
        self.depth = depth

        # window_nums = [[1, 12, 12],[1, 12, 12],[1, 6, 6], [1, 3, 3]]

        # down_scales = [2,2,2]


        group_size = group_size_tf

        self.group_size = group_size
        self.window_nums = window_nums_tf

        self.if_mamba = if_mamba
        self.depth_tf = depth
        if if_mamba:
            self.depth_tf = depth-1
            self.encoder_input = LIONLayer(dim, 1, window_shape, group_size_mamba, direction, False, operator, layer_id , n_layer)
            self.pos_emb_input = PositionEmbeddingLearned(input_channel=3, num_pos_feats=dim)
            # self.pos_emb_input = LBPEmbeddingLearned(3, dim, 8)
            self.downsample_input = PatchMerging3D(dim, dim, down_scale=down_scales[0], norm_layer=norm_fn)

        shift = [False, shift]
        for idx in range(1, self.depth):
            self.encoder.append(
                # win_and_group_selfatt(dim, 8, window_nums[idx], group_size[idx], operator,n_layer)
                win_and_group_tf(dim, 8, window_nums_tf[idx], group_size[idx], operator, n_layer)


            )
            # self.pos_emb_list.append(PositionEmbeddingLearned(input_channel=3, num_pos_feats=dim))
            self.pos_emb_list.append(LBPEmbeddingLearned(3, dim, 8))
            self.downsample_list.append(PatchMerging3D(dim, dim, down_scale=down_scales[idx], norm_layer=norm_fn))

        self.if_Middle_layer = if_Middle_layer
        if if_Middle_layer:
            # self.Middle_layer = Win_and_group_tf_Layer( dim, window_nums[self.depth], group_size[self.depth], operator, n_layer)
            self.Middle_layer = win_and_group_tf(dim, 8, window_nums_tf[self.depth], group_size[self.depth], operator, n_layer)
            # self.Middle_layer = LIONLayer(dim, 1, [13, 13, 4], group_size_mamba/4, direction, True, operator, layer_id, n_layer)
            # self.Middle_pos_emb = PositionEmbeddingLearned(input_channel=3, num_pos_feats=dim)
            self.Middle_pos_emb = LBPEmbeddingLearned(3, dim, 8)
            # self.Middle_upsample = PatchExpanding3D(dim)
            # self.Middle_norm = norm_fn(dim)

            ########################################
            # self.decoder_norm2 = nn.ModuleList()
        else:
            # self.Middle_pos_emb = PositionEmbeddingLearned(input_channel=3, num_pos_feats=dim)
            self.Middle_pos_emb = LBPEmbeddingLearned(3, dim, 8)

        self.decoder = nn.ModuleList()
        # self.decoder_norm = nn.ModuleList()
        self.upsample_list = nn.ModuleList()
        for idx in range(1, self.depth):
            self.decoder.append(
                # Win_and_group_tf_Layer( dim, window_nums[self.depth - idx-1], group_size[self.depth - idx-1], operator, n_layer)
                # win_and_group_crossatt(dim, 8, window_nums_tf[self.depth - idx-1], group_size[self.depth - idx-1], operator, n_layer)
                win_and_group_tf(dim, 8, window_nums_tf[self.depth - idx-1], group_size[self.depth - idx-1], operator, n_layer)
            )
            # self.decoder_norm.append(norm_fn(dim))
            self.upsample_list.append(PatchExpanding3D(dim))
            # if self.if_Middle_layer:
            #     self.decoder_norm2.append(norm_fn(dim))
        # self.out_norm = norm_fn(dim)


        if if_mamba:
            self.decoder_output = LIONLayer(dim, 1, window_shape, group_size_mamba, direction, False, operator, layer_id ,n_layer)
            # self.decoder_norm_output = norm_fn(dim)
            self.upsample_output = PatchExpanding3D(dim)
            # if self.if_Middle_layer:
            #     self.decoder_norm_output2 = norm_fn(dim)
        self.diffusion = diffusion
        if diffusion:
            self.Diffusion = PatchDiffusion3D(dim, conv=True)

    def forward(self, x):

        device = x.features.device
        # x_features_raw = x.features
        if self.if_mamba:
            pos_emb = self.get_pos_embed(spatial_shape=x.spatial_shape, coors=x.indices[:, 1:],embed_layer=self.pos_emb_input)
            # pos_emb = self.pos_emb_input(x)
            # x_features = x_features + x_pos_embed
            x = replace_feature(x, pos_emb + x.features)  # x + pos_emb
            x_input = self.encoder_input(x)
            x, unq_inv_input = self.downsample_input(x_input)



        features = []
        index = []
        mappings = []
        pos_embs = []

        for idx in range(self.depth_tf):
            # pos_emb = self.get_pos_embed(spatial_shape=x.spatial_shape, coors=x.indices[:, 1:],
            #                              embed_layer=self.pos_emb_list[idx])
            pos_emb = self.pos_emb_list[idx](x)
            pos_embs.append(pos_emb)
            x = replace_feature(x, pos_emb + x.features)  # x + pos_emb

            mapping = sprace_to_groupfearutes(self.group_size[idx], self.window_nums[idx], index=x.indices, batch_size=x.batch_size, serializ=self.serializ,
                                               sparse_shape=x.spatial_shape, dim=self.dim, device=device)
            mappings.append(mapping)

            x = self.encoder[idx](x, mapping)
            features.append(x)
            x, unq_inv = self.downsample_list[idx](x)
            index.append(unq_inv)

        if self.if_Middle_layer:
            depth = self.depth
            mapping = sprace_to_groupfearutes(self.group_size[depth], self.window_nums[depth], index=x.indices,
                                              batch_size=x.batch_size, serializ=self.serializ,
                                              sparse_shape=x.spatial_shape, dim=self.dim, device=device)
            pos_emb = self.Middle_pos_emb(x)
            x = replace_feature(x, pos_emb + x.features)  # x + pos_emb
            x = self.Middle_layer(x, mapping)
            # x = self.Middle_layer(x)
        else:
            depth = self.depth
            mapping = sprace_to_groupfearutes(self.group_size[depth], self.window_nums[depth], index=x.indices,
                                              batch_size=x.batch_size, serializ=self.serializ,
                                              sparse_shape=x.spatial_shape, dim=self.dim, device=device)
            pos_emb = self.Middle_pos_emb(x)
            x = replace_feature(x, pos_emb + x.features)
            mappings.append(mapping)

        depth = self.depth_tf-1
        for idx in range(self.depth_tf):
            if self.if_Middle_layer:
                x_up = features[depth-idx]
                unq_inv = index[depth - idx]
                mapping = mappings[depth - idx]
                pos_emb = pos_embs[depth - idx]
                x = self.upsample_list[idx](x, x_up, unq_inv)
                x = replace_feature(x, pos_emb + x.features)
                # x = self.decoder[idx](x, mapping, x_up)
                x = self.decoder[idx](x, mapping)
                # x = replace_feature(x, x.features)
            else:
                x_up = features[depth - idx]
                unq_inv = index[depth - idx]
                mapping = mappings[depth - idx+1]
                pos_emb = pos_embs[depth - idx]
                # x = self.decoder[idx](x, mapping, x_up)
                x = self.decoder[idx](x, mapping)
                x = self.upsample_list[idx](x, x_up, unq_inv)
                x = replace_feature(x, pos_emb + x.features)


        if self.if_mamba:
            if self.if_Middle_layer:
                x = self.upsample_output(x, x_input, unq_inv_input)
                # x = replace_feature(x, self.decoder_norm_output(x.features))
                x = self.decoder_output(x)
                # x = replace_feature(x, self.decoder_norm_output2(x.features))
            else:
                x = self.decoder_output(x)
                x = self.upsample_output(x, x_input, unq_inv_input)
                # x = replace_feature(x, self.decoder_norm_output(x.features))
        if self.diffusion:
            x = self.Diffusion(x)
        return x

    def get_pos_embed(self, spatial_shape, coors, embed_layer, normalize_pos=True):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''
        # [N,]
        window_shape = spatial_shape[::-1]  # spatial_shape:   win_z, win_y, win_x ---> win_x, win_y, win_z

        embed_layer = embed_layer
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            ndim = 2
            # win_x, win_y, win_z = window_shape
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        z, y, x = coors[:, 0] - win_z / 2, coors[:, 1] - win_y / 2, coors[:, 2] - win_x / 2

        if normalize_pos:
            if ndim == 2:
                x = x / win_x * 2 * 3.1415  # [-pi, pi]
                y = y / win_y * 2 * 3.1415  # [-pi, pi]
            else:
                x = x / win_x * 2 * 3.1415  # [-pi, pi]
                y = y / win_y * 2 * 3.1415  # [-pi, pi]
                z = z / win_z * 2 * 3.1415  # [-pi, pi]

        if ndim == 2:
            location = torch.stack((x, y, z), dim=-1)
        else:
            location = torch.stack((x, y, z), dim=-1)
        pos_embed = embed_layer(location.contiguous())

        return pos_embed



class MLPBlock(nn.Module):
    def __init__(self, input_channel, out_channel, norm_fn):
        super().__init__()
        self.mlp_layer = nn.Sequential(
            nn.Linear(input_channel, out_channel),
            norm_fn(out_channel),
            nn.GELU())

    def forward(self, x):
        mpl_feats = self.mlp_layer(x)
        return mpl_feats


# for waymo and nuscenes, kitti, once
class GPTF3DBackbone_vxmerging(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]
        norm_fn = partial(nn.LayerNorm)

        dim = model_cfg.FEATURE_DIM
        # num_layers = model_cfg.NUM_LAYERS
        depths = model_cfg.DEPTHS
        layer_down_scales = model_cfg.LAYER_DOWN_SCALES
        direction = model_cfg.DIRECTION
        diffusion = model_cfg.DIFFUSION
        shift = model_cfg.SHIFT
        diff_scale = model_cfg.DIFF_SCALE
        self.window_shape = model_cfg.WINDOW_SHAPE
        # self.window_shape_tf = model_cfg.WINDOW_SHAPE_TF
        self.group_size_mamba = model_cfg.GROUP_SIZE_MAMBA
        self.group_size_tf = model_cfg.GROUP_SIZE_TF
        self.layer_dim = model_cfg.LAYER_DIM
        self.linear_operator = model_cfg.OPERATOR
        self.window_nums_TF = model_cfg.window_nums_TF

        self.n_layer = len(depths) * depths[0] * 2 * 2 + 2

        self.if_mamba = model_cfg.IF_mamba
        self.if_Middle_layer = model_cfg.IF_Middle_layer

        self.serializ = model_cfg.SERIALIZE

        # down_scale_list = [[2, 2, 2],
        #                    [2, 2, 2],
        #                    [2, 2, 1],
        #                    [1, 1, 2],
        #                    [1, 1, 2]
        #                    ]
        # total_down_scale_list = [down_scale_list[0]]
        # for i in range(len(down_scale_list) - 1):
        #     tmp_dow_scale = [x * y for x, y in zip(total_down_scale_list[i], down_scale_list[i + 1])]
        #     total_down_scale_list.append(tmp_dow_scale)

        # assert num_layers == len(depths)
        assert len(layer_down_scales) == len(depths)
        # assert len(layer_down_scales[0]) == depths[0]
        # assert len(self.layer_dim) == len(depths)

        self.linear_1 = GPTFBlock(self.layer_dim[0], depths[0], layer_down_scales[0], self.window_shape[0],
                                  self.group_size_mamba[0], self.group_size_tf[0], direction, shift=shift, operator=self.linear_operator,
                                  window_nums_tf=self.window_nums_TF, if_mamba=self.if_mamba, if_Middle_layer=self.if_Middle_layer, serializ=self.serializ, diffusion=True)  ##[27, 27, 32] --》 [13, 13, 32]
        # self.norm1 = spconv.SparseSequential(norm_fn(self.layer_dim[0]),nn.GELU())
        self.dow1 = PatchMerging3D(self.layer_dim[0], self.layer_dim[0], down_scale=[1, 1, 2],
                                   norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)
        # self.dow1 = VoxelMerging3D(self.layer_dim[0], self.layer_dim[0], down_scale=[1, 1, 2],
        #                            norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)
        # [944, 944, 16] -> [472, 472, 8]

        self.linear_2 = GPTFBlock(self.layer_dim[1], depths[1], layer_down_scales[1], self.window_shape[1],
                                  self.group_size_mamba[0], self.group_size_tf[1], direction, shift=shift, operator=self.linear_operator,
                                  window_nums_tf = self.window_nums_TF, if_mamba = self.if_mamba, if_Middle_layer = self.if_Middle_layer, serializ=self.serializ, diffusion=True)
        self.dow2 = PatchMerging3D(self.layer_dim[1], self.layer_dim[1], down_scale=[1, 1, 2],
                                   norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        #  [236, 236, 8] -> [236, 236, 4]
        self.linear_3 = GPTFBlock(self.layer_dim[2], depths[2], layer_down_scales[2], self.window_shape[2],
                                  self.group_size_mamba[0], self.group_size_tf[2], direction, shift=shift, operator=self.linear_operator,
                                  window_nums_tf = self.window_nums_TF, if_mamba = self.if_mamba, if_Middle_layer = self.if_Middle_layer, serializ=self.serializ, diffusion=True)
        self.dow3 = PatchMerging3D(self.layer_dim[2], self.layer_dim[3], down_scale=[1, 1, 2],
                                   norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        #  [236, 236, 4] -> [236, 236, 2]
        # self.linear_4 = LIONBlock(self.layer_dim[3], depths[3], layer_down_scales[3], self.window_shape[3],
        #                           [7, 7], direction, shift=shift, operator=self.linear_operator,
        #                           )
        # self.dow4 = PatchMerging3D(self.layer_dim[3], self.layer_dim[3], down_scale=[1, 1, 2],
        #                            norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        self.linear_4 = GPTFBlock(self.layer_dim[3], depths[3], layer_down_scales[3], self.window_shape[3],
                                  self.group_size_mamba[1], self.group_size_tf[3], direction, shift=shift, operator=self.linear_operator,
                                  window_nums_tf = self.window_nums_TF, if_mamba = self.if_mamba, if_Middle_layer = self.if_Middle_layer, serializ=self.serializ, diffusion=True)
        self.dow4 = PatchMerging3D(self.layer_dim[2], self.layer_dim[3], down_scale=[1, 1, 2],
                                   norm_layer=norm_fn, diffusion=diffusion, diff_scale=diff_scale)

        # (self, dim:int, group_size_tf, operator=None, n_layer=0, window_nums=[[1, 18, 18], [1, 6, 6]], if_mamba = True, if_Middle_layer = True):

        # self.linear_out = TFLayer_out(self.layer_dim[3], group_size_tf = 8, window_nums=[1, 12, 12], if_mamba = True, if_Middle_layer = True)
        # self.linear_out = LIONLayer(self.layer_dim[3], 1, [13, 13, 2], 256, direction=['x', 'y'], shift=False, operator=self.linear_operator, layer_id=32, n_layer=self.n_layer)
        self.linear_out = GPTFBlock(self.layer_dim[3], depths[3], layer_down_scales[3], self.window_shape[3],
                                  self.group_size_mamba[1], self.group_size_tf[3], direction, shift=shift, operator=self.linear_operator,
                                  window_nums_tf=self.window_nums_TF, if_mamba = self.if_mamba, if_Middle_layer = self.if_Middle_layer, serializ=self.serializ, diffusion=False)

        self.num_point_features = dim

        self.backbone_channels = {
            'x_conv1': 128,
            'x_conv2': 128,
            'x_conv3': 128,
            'x_conv4': 128
        }



    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords'].float()
        batch_size = batch_dict['batch_size']

        # encode
        # addcuncum = torch.ones(voxel_coords.size()[0])
        # energy = torch.cumsum(addcuncum, dim=0)  # voxel在第几个group第几个值

        # code, order = encode(voxel_coords[:, 1:], batch=voxel_coords[:, 0], depth=16, order="hilbert")

        # show_voxel(voxel_coords[order][voxel_coords[:, 0] == 0][:, [2, 3, 1]].float().cpu(), energy[voxel_coords.cpu()[:, 0] == 0]) #

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        # ############################
        # o = []
        # count = 0
        # for i in range(x.batch_size):
        #     count += x.indices[:, 0][x.indices[:, 0] == i].shape[0]
        #     o.append(count)
        # o = torch.cuda.IntTensor(o)
        # self.nsample = 16
        # idx, diff = knnquery(self.nsample, x.indices[:, 1:].float(), x.indices[:, 1:].float(), o, o)
        # x_indices_n = x.indices[:,1:][[idx]]
        # x_indices_n0 = x_indices_n[:, 0:1, :]
        # x_indices_n1 = x_indices_n[:,1:,:]
        # x_indices_n2 = x_indices_n1 - x_indices_n0
        # ############################




        # x, voxel_query_num = self.getatt_spares(x, self.norm1)
        # voxel_query_num = getatt_spares(x)
        x = self.linear_1(x)
        # x = self.norm1(x)
        x1, _ = self.dow1(x)  ## 14.0k --> 16.9k  [32, 1000, 1000]-->[16, 1000, 1000]

        # x1, voxel_query_num1 = self.getatt_spares(x1, self.norm2)
        # voxel_query_num1 = getatt_spares(x1)
        x2 = self.linear_2(x1)
        # x2 = self.norm2(x2)
        x2, _ = self.dow2(x2)  ## 16.9k --> 18.8k  [16, 1000, 1000]-->[8, 1000, 1000]

        # x2, voxel_query_num2 = self.getatt_spares(x2, self.norm3)
        # voxel_query_num2 = getatt_spares(x2)
        x3 = self.linear_3(x2)
        # x3 = self.norm2(x3)
        x3, _ = self.dow3(x3)  ## 18.8k --> 19.1k  [8, 1000, 1000]-->[4, 1000, 1000]

        # x3, voxel_query_num3 = self.getatt_spares(x3, self.norm4)
        # voxel_query_num3 = getatt_spares(x3)
        x4 = self.linear_4(x3)
        # x4 = self.norm4(x4)
        x4, _ = self.dow4(x4)  ## 19.1k --> 18.5k  [4, 1000, 1000]-->[2, 1000, 1000]

        # x4, _ = self.getatt_spares(x4, self.norm_out)

        x = self.linear_out(x4)
        # x = self.norm_out(x)


        batch_dict.update({
            'encoded_spconv_tensor': x,
            'encoded_spconv_tensor_stride': 1
        })

        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x1,
                'x_conv2': x2,
                'x_conv3': x3,
                'x_conv4': x4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': torch.tensor([1, 1, 2], device=x1.features.device).float(),
                'x_conv2': torch.tensor([1, 1, 4], device=x1.features.device).float(),
                'x_conv3': torch.tensor([1, 1, 8], device=x1.features.device).float(),
                'x_conv4': torch.tensor([1, 1, 16], device=x1.features.device).float(),
            }
        })

        return batch_dict
