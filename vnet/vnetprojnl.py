import warnings
import torch
import torch.nn as nn

from .non_local import NONLocalBlock2D

from .blocks import *

__all__ = ('VNetProj')


class VNetProjBasenl(nn.Module):
    def __init__(self, in_channels, out_channels, n_frames, **kwargs):
        super(VNetProjBasenl, self).__init__()
        norm_type = nn.BatchNorm3d
        act_type = nn.ReLU
        se_type = None
        drop_type = None
        feats = [16, 32, 64, 128, 256]
        num_blocks = [1, 2, 3, 3]
        block_name = 'residual'
        self._use_aspp = False
        if 'norm_type' in kwargs.keys():
            norm_type = kwargs['norm_type']
        if 'act_type' in kwargs.keys():
            act_type = kwargs['act_type']
        if 'feats' in kwargs.keys():
            feats = kwargs['feats']
        if 'se_type' in kwargs.keys():
            se_type = kwargs['se_type']
        if 'num_blocks' in kwargs.keys():
            num_blocks = kwargs['num_blocks']
        if 'drop_type' in kwargs.keys():
            drop_type = kwargs['drop_type']
        if 'use_aspp' in kwargs.keys():
            self._use_aspp = kwargs['use_aspp']
        if 'block_name' in kwargs.keys():
            block_name = kwargs['block_name']

        self.n_frames = n_frames

        self.in_conv = InputBlock(in_channels, feats[0],
                                  norm_type=norm_type,
                                  act_type=act_type)

        self.nl = NONLocalBlock2D(feats[4], inter_channels=feats[4]//2)

        self.down1 = DownBlock(feats[0], feats[1], down_t=True, norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[0], block_name=block_name)
        self.down2 = DownBlock(feats[1], feats[2], down_t=False, norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[1], block_name=block_name)
        self.down3 = DownBlock(feats[2], feats[3], down_t=True, norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[2], block_name=block_name)
        self.down4 = DownBlock(feats[3], feats[4], down_t=False, norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[3], block_name=block_name)
        self.projin = nn.Sequential(
            nn.Conv3d(in_channels=feats[0], out_channels=feats[0], kernel_size=(n_frames, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2)),
            nn.ReLU()
        )
        self.proj1 = nn.Sequential(
            nn.Conv3d(in_channels=feats[1], out_channels=feats[1], kernel_size=(n_frames//2, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2)),
            nn.ReLU()
        )
        self.proj2 = nn.Sequential(
            nn.Conv3d(in_channels=feats[2], out_channels=feats[2], kernel_size=(n_frames//2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU()
        )
        self.proj3 = nn.Sequential(
            nn.Conv3d(in_channels=feats[3], out_channels=feats[3], kernel_size=(n_frames//4, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU()
        )
        self.proj4 = nn.Sequential(
            nn.Conv3d(in_channels=feats[4], out_channels=feats[4], kernel_size=(n_frames//4, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU()
        )

        if self._use_aspp:
            self.aspp = ASPP(feats[4], dilations=[1, 2, 3, 4], norm_type=norm_type, act_type=act_type,
                             drop_type=drop_type)

        self.up4 = UpBlock2d(feats[4], feats[3], feats[4], act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[3], block_name=block_name)
        self.up3 = UpBlock2d(feats[4], feats[2], feats[3], act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[2], block_name=block_name)
        self.up2 = UpBlock2d(feats[3], feats[1], feats[2], act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[1], block_name=block_name)
        self.up1 = UpBlock2d(feats[2], feats[0], feats[1], act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[0], block_name=block_name)

        self.out_block = OutBlock2d(feats[1], out_channels)

        # self.maxpool = nn.AdaptiveMaxPool3d((None, 1, None))

        init_weights(self)

    def forward(self, input):
        # if input.size(2) // 16 == 0 or input.size(3) // 16 == 0 or input.size(4) // 16 == 0:
        #     raise RuntimeError("input tensor shape is too small")
        input = self.in_conv(input)
        down1 = self.down1(input)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        if self._use_aspp:
            down4 = self.aspp(down4)

        projin = self.projin(input).squeeze(2).contiguous()
        proj1 = self.proj1(down1).squeeze(2).contiguous()
        proj2 = self.proj2(down2).squeeze(2).contiguous()
        proj3 = self.proj3(down3).squeeze(2).contiguous()
        proj4 = self.proj4(down4).squeeze(2).contiguous()

        up4 = self.up4(proj4, proj3)
        up4 = self.nl(up4)
        up3 = self.up3(up4, proj2)
        up2 = self.up2(up3, proj1)
        up1 = self.up1(up2, projin)

        out = self.out_block(up1)
        return out


class VNetProjnl(VNetProjBasenl):
    def __init__(self, in_channels, out_channels, n_frames, **kwargs):
        super(VNetProjnl, self).__init__(in_channels, out_channels, n_frames, **kwargs)
