from typing import Optional
import sys
# sys.path.append("/home/xiangcx/cb/code/")
from openfold.model.dropout import DropoutRowwise
from openfold.model.pair_transition import PairTransition
from openfold.model.triangular_attention import TriangleAttention
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    FusedTriangleMultiplicationIncoming,
    FusedTriangleMultiplicationOutgoing
)
from openfold.utils.tensor_utils import add
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops.layers.torch import Rearrange

def CrossEntropyLoss(pred, true):
    sum = torch.sum(torch.mul(true[0], torch.log(pred[0] + 1e-6)), dim=(-1))
    loss = -torch.nanmean(sum)
    return loss

class twoD_feats_Stack(nn.Module):
    def __init__(
            self,
            c_z: int,
            c_hidden_mul: int,
            c_hidden_pair_att: int,
            no_heads_pair: int,
            transition_n: int,
            pair_dropout: float,
            fuse_projection_weights: bool,
            inf: float,
    ):
        super(twoD_feats_Stack, self).__init__()

        if fuse_projection_weights:
            self.tri_mul_out = FusedTriangleMultiplicationOutgoing(
                c_z,
                c_hidden_mul,
            )
            self.tri_mul_in = FusedTriangleMultiplicationIncoming(
                c_z,
                c_hidden_mul,
            )
        else:
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                c_z,
                c_hidden_mul,
            )
            self.tri_mul_in = TriangleMultiplicationIncoming(
                c_z,
                c_hidden_mul,
            )

        self.tri_att_start = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)

    def forward(self,
                z: torch.Tensor,
                chunk_size: Optional[int] = None,
                use_lma: bool = False,
                inplace_safe: bool = False,
                _mask_trans: bool = True,
                _attn_chunk_size: Optional[int] = None
                ) -> torch.Tensor:

        if (_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        tmu_update = self.tri_mul_out(
            z,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if (not inplace_safe):
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        del tmu_update

        tmu_update = self.tri_mul_in(
            z,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if (not inplace_safe):
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        del tmu_update

        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_start(
                        z,
                        chunk_size=_attn_chunk_size,
                        use_memory_efficient_kernel=False,
                        use_lma=use_lma,
                        inplace_safe=inplace_safe,
                    )
                ),
                inplace=inplace_safe,
                )

        z = z.transpose(-2, -3)
        if (inplace_safe):
            z = z.contiguous()

        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_end(
                        z,
                        chunk_size=_attn_chunk_size,
                        use_memory_efficient_kernel=False,
                        use_lma=use_lma,
                        inplace_safe=inplace_safe,
                    )
                ),
                inplace=inplace_safe,
                )

        z = z.transpose(-2, -3)
        if (inplace_safe):
            z = z.contiguous()

        z = add(z,
                self.pair_transition(
                    z,  chunk_size=chunk_size,
                ),
                inplace=inplace_safe,
                )

        return z

class twoD_feats_Net(nn.Module): # 仿照默认设置
    def __init__(
            self,
            num_stacks:int):
        super(twoD_feats_Net, self).__init__()
        self.layers = nn.ModuleList([twoD_feats_Stack(
        c_z=128,
        c_hidden_mul = 128, # 128
        c_hidden_pair_att = 32, # 32
        no_heads_pair = 4, # 4
        transition_n = 4,
        pair_dropout = 0.25, # 0.25
        fuse_projection_weights = False,
        inf = 1e9
        ) for _ in range(num_stacks)])


    def forward(self,z) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z)
        return z

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.InstanceNorm2d(inplanes, affine=True)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation))
            bns.append(nn.InstanceNorm2d(width, affine=True))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.InstanceNorm2d(width * scale, affine=True)

        self.conv_st = nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1)

        self.relu = nn.ELU(inplace=True)
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.relu(self.bns[i](sp))
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        if self.stype == 'stage':
            residual = self.conv_st(residual)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out

class Res2Net(nn.Module):

    def __init__(self, in_channel, layers, baseWidth=26, scale=4):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(in_channel, affine=True),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channel, 64, 1),
        )
        self.layer1 = self._make_layer(Bottle2neck, 64, layers[0])
        self.layer2 = self._make_layer(Bottle2neck, 128, layers[1])
        self.layer3 = self._make_layer(Bottle2neck, 128, layers[2])
        self.layer4 = self._make_layer(Bottle2neck, 128, layers[3])

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride, stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        d = 1
        for i in range(1, blocks):
            d = 2 * d % 31
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale, dilation=d))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class DistPredictorMSA_V2(nn.Module):
    def __init__(self, in_channel=526, n_blocks=[3, 3, 9, 3]):
        super(DistPredictorMSA_V2, self).__init__()
        self.net = nn.Sequential(
            Res2Net(in_channel, n_blocks)
        )
        self.net2 = twoD_feats_Stack(
        c_z=512,
        c_hidden_mul = 128, # 128
        c_hidden_pair_att = 32, # 32
        no_heads_pair = 4, # 4
        transition_n = 4,
        pair_dropout = 0.25, # 0.25
        fuse_projection_weights = False,
        inf = 1e9
        )
        self.out_elu = nn.Sequential(
            nn.InstanceNorm2d(512, affine=True),
            nn.ELU(inplace=True)
        )
        self.conv_d = nn.Conv2d(512, 37, 1)
        self.conv_p = nn.Conv2d(512, 13, 1)
        self.conv_t = nn.Conv2d(512, 25, 1)
        self.conv_o = nn.Conv2d(512, 25, 1)

    def forward(self, msa,return_logits=False):
        with torch.no_grad():
            f2d = self.get_f2d(msa)
        output_tensor = self.net(f2d)  # 1,512,L,L
        output_tensor = Rearrange('N C H W-> N H W C')(output_tensor)  # (1, L, L, c_z)
        pred_logits = self.net2(output_tensor)  # (1, dim, L, L); L: sequence length
        pred_logits = Rearrange('N H W C -> N C H W')(pred_logits)
        output_tensor = self.out_elu(pred_logits)
        symm = output_tensor + output_tensor.permute(0, 1, 3, 2)
        pred_distograms = {}
        pred_distograms['dist'] = F.softmax(self.conv_d(symm), dim=1).permute(0, 2, 3, 1)  # 1,L,L,37
        pred_distograms['omega'] = F.softmax(self.conv_o(symm), dim=1).permute(0, 2, 3, 1)
        pred_distograms['theta'] = F.softmax(self.conv_t(output_tensor), dim=1).permute(0, 2, 3, 1)
        pred_distograms['phi'] = F.softmax(self.conv_p(output_tensor), dim=1).permute(0, 2, 3, 1)
        if return_logits:
            return pred_distograms, output_tensor
        else:
            return pred_distograms

    def get_f2d(self, msa):
        """ calculate features from MSA """
        device = msa.device
        nrow, ncol = msa.size()[-2:]
        msa1hot = (torch.arange(21, device=device) == msa[..., None]).float()
        w = self.reweight(msa1hot, .8)
        # print(w.shape)
        # print(msa1hot.shape)
        # 1D features
        f1d_seq = msa1hot[0, :, :20]
        f1d_pssm = self.msa2pssm(msa1hot, w)

        f1d = torch.cat([f1d_seq, f1d_pssm], dim=1)

        # 2D features
        f2d_dca = self.fast_dca(msa1hot, w) if nrow > 1 else torch.zeros([ncol, ncol, 442], device=device)

        # concat
        f2d = torch.cat([f1d[:, None, :].repeat([1, ncol, 1]),
                         f1d[None, :, :].repeat([ncol, 1, 1]),
                         f2d_dca], dim=-1)
        f2d = Rearrange('(N H) W C->N C H W', N=1)(f2d)  # HWC->NCHW
        return f2d

    @staticmethod
    def msa2pssm(msa1hot, w):
        """PSSM: position-specific scoring matrix"""
        beff = w.sum()
        f_i = (w[:, None, None] * msa1hot).sum(dim=0) / beff + 1e-9
        h_i = (-f_i * torch.log(f_i)).sum(dim=1)
        return torch.cat([f_i, h_i[:, None]], dim=1)

    @staticmethod
    def reweight(msa1hot, cutoff):
        """
        calculate weight of each homologous sequences
        the more similar sequences there are, the lower weight the sequence is.
        """

        id_min = msa1hot.size(1) * cutoff
        id_mtx = torch.tensordot(msa1hot, msa1hot, [[1, 2], [1, 2]])
        id_mask = id_mtx > id_min
        w = 1.0 / id_mask.sum(dim=-1)
        return w

    @staticmethod
    def fast_dca(msa1hot, weights, penalty=4.5):
        """
        see "input features" in Methods section of paper for details
        """

        device = msa1hot.device
        nr, nc, ns = msa1hot.size()
        try:
            x = msa1hot.view(nr, nc * ns)
        except RuntimeError:
            x = msa1hot.contiguous().view(nr, nc * ns)
        num_points = weights.sum() - torch.sqrt(weights.mean())
        mean = (x * weights[:, None]).sum(dim=0, keepdims=True) / num_points
        x = (x - mean) * torch.sqrt(weights[:, None])
        cov = torch.matmul(x.permute(1, 0), x) / num_points

        cov_reg = cov + torch.eye(nc * ns, device=device) * penalty / torch.sqrt(weights.sum())
        inv_cov = torch.inverse(cov_reg)

        x1 = inv_cov.view(nc, ns, nc, ns)
        x2 = x1.permute(0, 2, 1, 3)
        features = x2.reshape(nc, nc, ns * ns)
        nc_eye = torch.eye(nc, device=device)
        x3 = torch.sqrt(torch.square(x1[:, :-1, :, :-1]).sum((1, 3))) * (1 - nc_eye)
        apc = x3.sum(dim=0, keepdims=True) * x3.sum(dim=1, keepdims=True) / x3.sum()
        contacts = (x3 - apc) * (1 - nc_eye)

        return torch.cat([features, contacts[:, :, None]], dim=2)


