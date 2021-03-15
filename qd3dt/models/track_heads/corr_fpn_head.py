import torch
import torch.nn as nn
import torch.nn.functional as F
# from spatial_correlation_sampler import SpatialCorrelationSampler as corr
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class CorrFPNHead(nn.Module):

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 num_levels=4,
                 corr_params=dict(
                     patch_size=17,
                     kernel_size=1,
                     padding=0,
                     stride=1,
                     dilation_patch=1)):
        super(CorrFPNHead, self).__init__()
        # init self params
        self.in_channels = in_channels
        self.out_channels = out_channels
        # general configs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.num_levels = num_levels

        self.corr_params = corr_params
        if self.corr_params:
            self.corr = corr(**corr_params)
            corr_channels = corr_params['patch_size']**2
        else:
            corr_channels = 0

        self.map_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()

        for i in range(num_levels):
            concat_channels = corr_channels + in_channels * 2
            self.map_convs.append(
                ConvModule(
                    concat_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                    inplace=False))
            self.out_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activate_last=self.activation,
                    inplace=False))

    def extract_corr_feats(self, x, ref_x):
        corr_x = [self.corr(_x, _ref_x) for _x, _ref_x in zip(x, ref_x)]
        corr_x = [
            _corr_x.view(
                _corr_x.size(0), -1, _corr_x.size(3), _corr_x.size(4))
            for _corr_x in corr_x
        ]
        # c = 17 * 17 + 256 + 256 = 801
        track_x = [torch.cat(xs, dim=1) for xs in zip(x, ref_x, corr_x)]
        return track_x

    def forward(self, x, ref_x):
        x = x[:self.num_levels]
        ref_x = ref_x[:self.num_levels]

        if hasattr(self, 'corr'):
            corr_x = self.extract_corr_feats(x, ref_x)
        else:
            corr_x = [torch.cat(xs, dim=1) for xs in zip(x, ref_x)]

        map_feats = [
            map_conv(corr_x[i]) for i, map_conv in enumerate(self.map_convs)
        ]

        for i in range(self.num_levels - 1, 0, -1):
            map_feats[i - 1] += F.interpolate(
                map_feats[i], scale_factor=2, mode='nearest')

        outs = [
            self.out_convs[i](map_feats[i]) for i in range(self.num_levels)
        ]

        return outs
