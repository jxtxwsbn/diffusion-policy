from typing import Dict
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import diffusion_policy.model.vision.crop_randomizer as dmvc
from collections import OrderedDict

class ResBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size):
        super(ResBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            # nn.GroupNorm(hidden_dim//16, hidden_dim),
            nn.ReLU(hidden_dim)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            # nn.GroupNorm(hidden_dim // 16, hidden_dim),
        )
        self.relu = nn.ReLU(hidden_dim)
        self.rescale_channel = None
        if input_channels != hidden_dim:
            self.rescale_channel = nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            )

    def forward(self, xx):
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.rescale_channel:
            out += self.rescale_channel(residual)
        else:
            out += residual
        out = self.relu(out)
        return out


class ResUNet(torch.nn.Module):
    def __init__(self, n_input_channel=3, n_output_channel=32, n_hidden=32, kernel_size=3,
                 input_shape=(3, 96, 96), h=80, w=80, fixed_crop=False, local_cond_type=''):
        super().__init__()
        self.hid = n_hidden
        self.n_input_channel = n_input_channel
        self.n_output_channel = n_output_channel
        self.n_neck_channel = 8 * self.hid
        self.fixed_crop = fixed_crop
        self.local_cond_type = local_cond_type
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.register_buffer('h', torch.tensor(h))
        self.register_buffer('w', torch.tensor(w))
        self.build()

    def build(self):
        
        # self.crop = dmvc.CropRandomizer(
        #                                 input_shape=self.input_shape,
        #                                 crop_height=self.h.item(),
        #                                 crop_width=self.w.item(),
        #                                 fixed_crop=self.fixed_crop
        #                             )
        
        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-0', nn.Conv2d(self.n_input_channel, self.hid, kernel_size=self.kernel_size, padding=1)),
            ('enc-e2relu-0', nn.ReLU()),
            ('enc-e2res-1', ResBlock(self.hid, self.hid, kernel_size=self.kernel_size)),
        ]))

        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-2', nn.MaxPool2d(2)),
            ('enc-e2res-2', ResBlock(self.hid, 2 * self.hid, kernel_size=self.kernel_size)),
        ]))
        self.conv_down_4 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-3', nn.MaxPool2d(2)),
            ('enc-e2res-3', ResBlock(2 * self.hid, 4 * self.hid, kernel_size=self.kernel_size)),
        ]))
        self.conv_down_8 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-4', nn.MaxPool2d(2)),
            ('enc-e2res-4', ResBlock(4 * self.hid, 8 * self.hid, kernel_size=self.kernel_size)),
        ]))
        self.conv_down_16 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-5', nn.MaxPool2d(2)),
            ('enc-e2res-5', ResBlock(8 * self.hid, 8 * self.hid, kernel_size=self.kernel_size)),
        ]))

        self.conv_up_8 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-1', ResBlock(16 * self.hid, 4 * self.hid, kernel_size=self.kernel_size)),
        ]))
        self.conv_up_4 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-2', ResBlock(8 * self.hid, 2 * self.hid, kernel_size=self.kernel_size)),
        ]))
        self.conv_up_2 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-3', ResBlock(4 * self.hid, 1 * self.hid, kernel_size=self.kernel_size)),
        ]))
        self.conv_up_1 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-4', ResBlock(2 * self.hid, 1 * self.hid, kernel_size=self.kernel_size)),
            ('dec-e2conv-4', nn.Conv2d(1 * self.hid, self.n_output_channel, kernel_size=self.kernel_size, padding=1)),
        ]))

        self.upsample_16_8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_8_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_4_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.state_lin = nn.Linear(2, self.n_output_channel)
        self.trajactory_lin = nn.Linear(2, self.n_output_channel) if self.local_cond_type.find('input') > -1 \
            else None

        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Free parameters: ', self.total_params)

        self.activation = nn.ReLU()

    def forwardEncoder(self, obs):
        feature_map_1 = self.conv_down_1(obs)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        # print(feature_map_1.shape,feature_map_2.shape, feature_map_4.shape, feature_map_8.shape,feature_map_16.shape)
        return feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16

    def forwardDecoder(self, feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16):
        concat_8 = torch.cat((feature_map_8, self.upsample_16_8(feature_map_16)), dim=1)
        feature_map_up_8 = self.conv_up_8(concat_8)

        concat_4 = torch.cat((feature_map_4, self.upsample_8_4(feature_map_up_8)), dim=1)
        feature_map_up_4 = self.conv_up_4(concat_4)

        concat_2 = torch.cat((feature_map_2, self.upsample_4_2(feature_map_up_4)), dim=1)
        feature_map_up_2 = self.conv_up_2(concat_2)

        concat_1 = torch.cat((feature_map_1, self.upsample_2_1(feature_map_up_2)), dim=1)
        feature_map_up_1 = self.conv_up_1(concat_1)

        return feature_map_up_1

    def forward(self, obs=None, x=None):
        
        if obs is not None:
            img, state = obs['image'], obs['agent_pos']
        else:
            img = x
        # Visualizing obs
        # plt.figure()
        # plt.imshow(img[0].permute(1, 2, 0).clone().detach().cpu().numpy())
        # pos = state[0].clone().detach().cpu().numpy()
        # # pos[1] = -pos[1]
        # pos += 1
        # pos /= 2
        # pos *= 95
        # plt.scatter(pos[0], pos[1], c='r')
        # plt.show()
        # img = self.crop(img)
        # print(img.shape)
        feature_maps = self.forwardEncoder(img)
        global_info = None
        if self.local_cond_type.find('global'):
            global_info = feature_maps[-1].amax(dim=(-1,-2))
        feature_maps = self.forwardDecoder(*feature_maps)
        # feature_maps += self.state_lin(state).unsqueeze(2).unsqueeze(3)
        # feature_maps = self.activation(feature_maps)
        return feature_maps, global_info

    def extrac_feature(self, embedding, trajectory):
        '''

        Parameters
        ----------
        embedding: image embedding in shape [(b To) c h w]
        trajectory: trajectory in shape [b k 2]

        Returns
        -------
        feature: feature in shape [b k (To c)]
        '''
        assert embedding.shape[2] == self.h
        assert embedding.shape[3] == self.w
        bs = trajectory.shape[0]
        To = embedding.shape[0] // bs
        embedding = rearrange(embedding, '(bs To) c h w -> bs (To c) (h w)', bs=bs)

        traj_w_h = trajectory.reshape(-1, 2).clone()
        traj_w_h *= (self.input_shape[1] - 1) / 2
        traj_w_h += (self.h - 1) / 2
        traj_w_h = torch.round(traj_w_h)

        idxs = torch.clamp(traj_w_h, 0, self.h - 1)
        idxs = idxs[:, 0] + idxs[:, 1] * self.w
        idxs = idxs.reshape(bs, -1).long()

        features = []
        for i in range(bs):
            features.append(embedding[i:i+1, :, idxs[i]])
        features = torch.cat(features).permute(0, 2, 1)

        if self.trajactory_lin is not None:
            trajectory_lin_project = self.trajactory_lin(trajectory.reshape(-1, 2))
            features += trajectory_lin_project.reshape(bs, -1, self.n_output_channel).repeat(1, 1, To)

        return features, traj_w_h

# unet = ResUNet(n_input_channel=6, n_output_channel=16, n_hidden=32,kernel_size=3)
# x = torch.rand(1,6, 96, 96)
# y, g= unet(x=x)
# print(y.shape, g.shape)




from guided_diffusion.unet import UNetModel

class Unet2D(nn.Module):
    def __init__(self, 
                 image_size: tuple=(96,96),
                 n_input_channel=6,
                 num_input_frame=1,
                 n_output_channel=16
                 ):
        super(Unet2D, self).__init__()
        self.num_channel = n_input_channel
        self.num_input_frame = num_input_frame
        # self.num_output_frame = num_output_frame

        self.unet = UNetModel(
            image_size=image_size,
            in_channels=self.num_channel,
            model_channels=128,
            out_channels=n_output_channel,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=False,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )

    def forward(self, obs=None, x=None, task_embed=None, **kwargs):
        
        if obs is not None:
            x, state = obs['image'], obs['agent_pos']
        t = torch.zeros(x.shape[0]).to(x.device)
        # fc = x.shape[1]
        # x_cond = rearrange(x[:, fc-self.num_channel*self.num_input_frame:], 'b (f c) h w -> b c f h w', f=self.num_input_frame)
        # x = rearrange(x[:, :fc-self.num_channel*self.num_input_frame], 'b (f c) h w -> b c f h w', f=self.num_output_frame)
        # x = torch.cat([x, x_cond], dim=1)
        # print(x.shape)
        x = rearrange(x, 'b (f c) h w -> b c f h w', f=self.num_input_frame)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w'), None

# unet = Unet2D().cuda()
# x = torch.rand(24, 6, 96, 96).cuda()
# y = unet(x)
# print(y.shape)

# y = unet.unet(x,timesteps=torch.zeros(x.shape[0]),y=None)
# print(y.shape)


