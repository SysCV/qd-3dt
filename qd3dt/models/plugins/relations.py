import torch
import torch.nn as nn

from ..utils import ConvModule
from qd3dt.core import bbox_overlaps


class Relations(nn.Module):

    def __init__(self,
                 in_channels=1024,
                 inter_channels=1024,
                 groups=16,
                 num_embed_convs=1,
                 share_embed_convs=True,
                 with_loc=True):
        super(Relations, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.inter_channels = inter_channels
        assert not in_channels % groups
        self.num_embed_convs = num_embed_convs
        self.share_embed_convs = share_embed_convs
        self.with_loc = with_loc

        self.init_embed_convs()
        self.conv_out = ConvModule(
            self.inter_channels * self.groups,
            self.in_channels,
            kernel_size=1,
            activation=None,
            groups=self.groups)

    def init_embed_convs(self):
        self.embed_convs = nn.ModuleList()
        if not self.share_embed_convs:
            self.ref_embed_convs = nn.ModuleList()
        for i in range(self.num_embed_convs):
            in_channels = self.in_channels if i == 0 else self.inter_channels
            self.embed_convs.append(
                ConvModule(
                    in_channels,
                    self.inter_channels,
                    kernel_size=1,
                    activation='relu',
                    activate_last=False,
                    inplace=False))
            self.embed_convs.append(
                ConvModule(
                    in_channels,
                    self.inter_channels,
                    kernel_size=1,
                    activation='relu',
                    activate_last=False))
            if not self.share_embed_convs:
                self.ref_embed_convs.append(
                    ConvModule(
                        in_channels,
                        self.inter_channels,
                        kernel_size=1,
                        activation='relu',
                        activate_last=False,
                        inplace=False))
                self.ref_embed_convs.append(
                    ConvModule(
                        in_channels,
                        self.inter_channels,
                        kernel_size=1,
                        activation='relu',
                        activate_last=False))

    def forward(self, in_x, rois, in_ref_x=None, ref_rois=None):
        # x: [N_0, C]      ref_x: [N_1, C]
        # rois: [N_0, 4]   ref_rois: [N_1, 4]
        if in_ref_x is None:
            in_ref_x = in_x
            ref_rois = rois
        N_0, C = in_x.shape
        N_1, C_1 = in_ref_x.shape
        assert C == C_1
        x = in_x.view(N_0, C, 1, 1)
        ref_x = in_ref_x.view(N_0, C, 1, 1)

        for i, embed_conv in enumerate(self.embed_convs):
            x = embed_conv(x)
            if not self.share_embed_convs:
                ref_x = self.ref_embed_convs[i](ref_x)
            else:
                ref_x = embed_conv(ref_x)

        # [N, G, C // G]
        x = x.view(N_0, self.groups, -1)
        ref_x = ref_x.view(N_1, self.groups, -1)
        # [G, N_0, C // G]
        x = x.permute(1, 0, 2)
        # [G, C // G, N_1]
        ref_x = ref_x.permute(1, 2, 0)
        # [G, N_0, N_1]
        matrix = torch.matmul(x, ref_x)
        matrix /= x.shape[-1]**0.5
        # [N_0, G, N_1]
        matrix = matrix.permute(1, 0, 2)

        if self.with_loc:
            # [N_0, N_1]
            ious = bbox_overlaps(rois[:, 1:], ref_rois[:, 1:])
            ious = ious.view(N_0, 1, N_1).expand(N_0, self.groups, N_1)
            matrix += torch.log(ious + 1e-6)

        # [N_0, G, N_1]
        matrix = matrix.softmax(dim=2)
        # [N_0 * G, N_1]
        matrix = matrix.view(-1, N_1)
        # [N_0 * G, C] = [N_0 * G, N_1] * [N_1, C]
        y = torch.matmul(matrix, in_ref_x)
        # [N_0, C * G]
        y = y.view(N_0, -1, 1, 1)
        # [N_0, C]
        y = self.conv_out(y).view(N_0, -1)
        return y
