from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import copy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from unet import ResUNet
# from unet import Unet2D
from utils import visualize_pusht_images_sequnece, pos2pixel, pixel2map, pixel2pos, pix2xy, xy2pix, transposerc
import torchvision.transforms.functional as torchvisionf
import numpy as np
import einops

class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            use_pos_map=False,
            # use_pos_val=False,
            trans_aug = False,
            rot_aug = False,
            relative=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)
        print('n_obs_steps', n_obs_steps)
        print('horizons', horizon)


        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.trans_aug = trans_aug
        self.rot_aug = rot_aug
        self.use_pos_map = use_pos_map
        self.relative = relative
        self.kwargs = kwargs


        if self.use_pos_map:
            channel = 3+1
        else:
            channel = 3
        self.channel = channel
        self.obs_encoder = ResUNet(n_input_channel=n_obs_steps*channel, n_output_channel=horizon, n_hidden=48)
        # self.obs_encoder = Unet2D(n_input_channel=n_obs_steps*channel, n_output_channel=horizon)


        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        # nobs = self.normalizer.normalize(obs_dict)
        nobs = copy.deepcopy(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        # print('value', value.shape)
        T = self.horizon
        Da = self.action_dim
        # Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            
            imge_shape = nobs['image'].shape
            batch_size = imge_shape[0]
            # print('image shape',imge_shape)
            # visualize_pusht_images_sequnece(nobs['image'][:16,0,:,:].cpu(),word='test')
            # visualize_pusht_images_sequnece(nobs['image'][:16,1,:,:].cpu(),word='test')

            nobs['image'] = nobs['image'][:,:self.n_obs_steps,...].reshape(imge_shape[0], self.n_obs_steps*imge_shape[-3], *imge_shape[-2:])
            nobs['agent_pos'] = nobs['agent_pos'][:,:self.n_obs_steps,...].reshape(batch_size, self.n_obs_steps, self.action_dim)
            agent_pos_pix = pos2pixel(nobs['agent_pos'])
            # print(agent_pos_pix.shape)

            if self.relative:
                #------------------
                crop_size = imge_shape[-1]
                center_img = F.pad(nobs['image'],pad=(imge_shape[-1]//2,imge_shape[-1]//2, imge_shape[-2]//2, imge_shape[-2]//2))
                center_img = rearrange(center_img, 'b (o n) h w -> b o n h w', o=self.n_obs_steps)
                bs, to, c, h, w = center_img.shape
                # print(bs,to,crop_size)
                cano_imgs = []
                for i in range(bs):
                    for j in range(to):
                        top_row = agent_pos_pix[i,j,0]
                        left_col = agent_pos_pix[i,j,1]
                        img = center_img[i,j]
                        img = img[:, top_row.item() : top_row.item()+crop_size, left_col.item() : left_col.item() + crop_size]
                        # import matplotlib.pyplot as plt
                        # print(img.shape)
                        # print(top_row, left_col)
                        # plt.imshow(img.cpu().permute(1,2,0).numpy())
                        # plt.show()
                        img = img.unsqueeze(dim=0)
                        cano_imgs.append(img)
                cano_imgs = torch.cat(cano_imgs,dim=0)
                cano_imgs = rearrange(cano_imgs, '(b o) n h w -> b (o n) h w', b=bs)
                
                base_agent_pos_pix = agent_pos_pix[:,0:1,:].clone()
                base_agent_pos_pix = base_agent_pos_pix.repeat(1,self.horizon,1)
                # cano_label_pix = label_pix - base_agent_pos_pix + crop_size//2
                cano_agent_pos_pix = agent_pos_pix - agent_pos_pix[:,0:1,:]
                cano_agent_pos_pix = -cano_agent_pos_pix + crop_size//2

                nobs['image'] = cano_imgs
                agent_pos_pix = cano_agent_pos_pix

            if self.use_pos_map:
                agent_pos_map = pixel2map(agent_pos_pix.reshape(batch_size*self.n_obs_steps, self.action_dim))
                agent_pos_map = agent_pos_map.reshape(batch_size,self.n_obs_steps,imge_shape[-2], imge_shape[-1])
                # print(agent_pos_map.shape)
                # for i in range(10):
                #     print(agent_pos_map[i,0].sum())
                #     print(agent_pos_map[i,0,agent_pos_pix[i,0,0],agent_pos_pix[i,0,1]])
                agent_pos_map = agent_pos_map.unsqueeze(dim=-3)
                # print(agent_pos_map.shape)
                nobs['image'] = nobs['image'].reshape(batch_size, self.n_obs_steps, imge_shape[-3], *imge_shape[-2:])
                nobs['image'] = torch.cat((nobs['image'],agent_pos_map),dim=-3)
                nobs['image'] = nobs['image'].reshape(batch_size, self.n_obs_steps*self.channel, *imge_shape[-2:])
                # print(nobs['image'].shape)
                #         
            
            pre_action_map, g = self.obs_encoder(obs=nobs)
            pre_action_map_shape = pre_action_map.shape
            # print('action map shape',pre_action_map_shape)
            # pre_action_map2 = pre_action_map
        
        # inference action
        # pre_action_map = pre_action_map.reshape(-1,pre_action_map_shape[-1]*pre_action_map_shape[-2])
        pre_action_map = rearrange(pre_action_map, 'b t h w -> (b t) (h w)')
        best_action_index = torch.max(pre_action_map,dim=1,keepdim=False)
        # print(torch.max(pre_action_map[0]))
        # print('max value',best_action_index[0][0])
        best_action_index = best_action_index[1]
        # action_pre_x = best_action_index // pre_action_map_shape[-1]
        action_pre_x = torch.div(best_action_index,pre_action_map_shape[-1],rounding_mode='floor')
        action_pre_y = best_action_index % pre_action_map_shape[-1]
        action_pred = torch.stack((action_pre_x, action_pre_y),dim=1)
        # print(action_pred.shape)
        # action_pred = action_pred.reshape(pre_action_map_shape[0], pre_action_map_shape[1], action_dim)
        action_pred = rearrange(action_pred, '(b t) d -> b t d', b=pre_action_map_shape[0])
        
        if self.relative:
            # cano_label_pix = label_pix - base_agent_pos_pix + crop_size//2
            action_pred = (action_pred - crop_size//2) + base_agent_pos_pix
            action_pred = action_pred.clamp(min=0,max=95)
        # print('action pred',action_pred.shape)
        # visualize_pusht_images_sequnece(pre_action_map2.cpu().unsqueeze(dim=2)[0], action=action_pred.cpu()[0],softmax=False)
        # visualize_pusht_images_sequnece(pre_action_map2.cpu().unsqueeze(dim=2)[0], action=action_pred.cpu()[0],softmax=True)
        
        # from pixel action to space action
        action_pred = transposerc(action_pred)
        action_pred = pixel2pos(action_pred)
        

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    
    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        # print(batch['obs']['image'][0].shape)
        # print('***********************')
        # # print('agent_pose', batch['obs']['agent_pos'].shape)
        # # print(batch['obs']['agent_pos'][0,:,0])
        # # print('agent_action',batch['action'].shape)
        # # print(batch['action'][0,:,0])
        # print('***********************')
        
        vis= False
        if vis:
            pos_pix=pos2pixel(batch['obs']['agent_pos'][0])
            # print(pos_pix,'=====')
            action_pix = pos2pixel(batch['action'][0])
            # action_map = pixel2map(action_pix)
            # print(action_pix)
            # print(action_map)
            # print('trans aug',self.trans_aug)
            # print('rot aug', self.rot_aug)

            # visualize_pusht_images_sequnece(batch['obs']['image'][0].clone().cpu(), pos=pos_pix.cpu(), action=action_pix.cpu(), title='before normalization')
            
        
        # nobs = self.normalizer.normalize(batch['obs'])
        # nactions = self.normalizer['action'].normalize(batch['action'])

        nobs = copy.deepcopy(batch['obs'])         
        nactions = copy.deepcopy(batch['action'])
                
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            # this_nobs = dict_apply(nobs, 
            #     lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:])) # 64 x 2 x 3 x 96 x 96 --> 128 * 3 * 96 * 96
            # print(this_nobs['image'].shape,'########################')
            # print(this_nobs['agent_pos'].shape,'#######################')
             
            imge_shape = nobs['image'].shape
            agent_pos_shape = nobs['agent_pos'].shape
            nobs['image'] = nobs['image'][:,:self.n_obs_steps,...].reshape(batch_size, self.n_obs_steps*imge_shape[-3], *imge_shape[-2:])
            nobs['agent_pos'] = nobs['agent_pos'][:,:self.n_obs_steps,...].reshape(batch_size, self.n_obs_steps, agent_pos_shape[-1])
            label_pix = pos2pixel(nactions)
            agent_pos_pix = pos2pixel(nobs['agent_pos'])

            label_pix = transposerc(label_pix)
            agent_pos_pix = transposerc(agent_pos_pix)

            aug_step_max = 0
            if self.trans_aug:
                aug_step_max += 10
                
            if self.rot_aug:
                aug_step_max += 10
            
            
            if aug_step_max >0:
                aug_step = 0
                while True:
                    agent_pos_pix_new = agent_pos_pix.clone()
                    label_pix_new = label_pix.clone()
                    tran_x = 0
                    tran_y = 0
                    theta_degree = 0

                    if self.rot_aug:
                        # sample transformation matrix
                        theta_sigma = 2 * np.pi / 6
                        theta = np.random.normal(0, theta_sigma)
                        # print(theta/np.pi * 180)
                        rotm = np.array([[np.cos(-theta),-np.sin(-theta)],
                                         [np.sin(-theta), np.cos(-theta)]])
                        rotm = torch.from_numpy(rotm).float().to(label_pix.device)
                        theta_degree = (theta * 180)/np.pi
                        # rotate agent pos
                        pos_xy = pix2xy(agent_pos_pix_new)
                        pos_xy = torch.einsum('bij, jk -> bik', pos_xy, rotm.T)
                        agent_pos_pix_new = xy2pix(pos_xy)

                        # rotat action label
                        action_xy = pix2xy(label_pix_new)
                        action_xy = torch.einsum('bij, jk -> bik', action_xy, rotm.T)
                        label_pix_new = xy2pix(action_xy)


                    if self.trans_aug:
                        
                        # translation
                        rand = np.random.random()
                        
                        if rand<0.7:
                            trans_amount_x = 20
                            trans_amount_y = 20
                            tran_x = np.random.randint(low=0, high=trans_amount_x)-(trans_amount_x/2)
                            tran_y = np.random.randint(low=0, high=trans_amount_y)-(trans_amount_y/2)
                        
                        else:
                            trans_sigma = imge_shape[-1]/6
                            trans = np.random.normal(0, trans_sigma, size=2)
                            tran_x = np.round(trans[0])
                            tran_y = np.round(trans[1])

                    
                    # agent_pos_pix_new[:,:,0] = agent_pos_pix_new[:,:,0] + tran_x
                    # agent_pos_pix_new[:,:,1] = agent_pos_pix_new[:,:,1] + tran_y

                    agent_pos_pix_new[:,:,0] = agent_pos_pix_new[:,:,0] + tran_y
                    agent_pos_pix_new[:,:,1] = agent_pos_pix_new[:,:,1] + tran_x

                    label_pix_new[:,:,0] = label_pix_new[:,:,0] + tran_x
                    label_pix_new[:,:,1] = label_pix_new[:,:,1] + tran_y

                    pos_action_pix = torch.concat((agent_pos_pix_new,label_pix_new),dim=1)
                    # print(pos_action_pix.shape)
                    min_row_index = torch.min(pos_action_pix[:,:,0])
                    max_row_index = torch.max(pos_action_pix[:,:,0])
                    min_col_index = torch.min(pos_action_pix[:,:,1])
                    max_col_index = torch.max(pos_action_pix[:,:,1])
                    valid = (min_row_index>=0) & (min_col_index>=0) & (max_row_index<imge_shape[-1]) & (max_col_index<imge_shape[-2])
                    if valid:
                        break
                    aug_step +=1

                    if aug_step==aug_step_max:
                        break
                if valid:
                    label_pix = label_pix_new
                    agent_pos_pix = agent_pos_pix_new
                    # clockwise direction
                    nobs['image'] = torchvisionf.affine(nobs['image'], angle=theta_degree, translate=[tran_x,tran_y], scale=1, shear=0)

            
            if self.relative:
                #------------------
                crop_size = imge_shape[-1]
                center_img = F.pad(nobs['image'],pad=(imge_shape[-1]//2,imge_shape[-1]//2, imge_shape[-2]//2, imge_shape[-2]//2))
                center_img = rearrange(center_img, 'b (o n) h w -> b o n h w', o=self.n_obs_steps)
                bs, to, c, h, w = center_img.shape
                # print(bs,to,crop_size)
                cano_imgs = []
                for i in range(bs):
                    for j in range(to):
                        top_row = agent_pos_pix[i,j,0]
                        left_col = agent_pos_pix[i,j,1]
                        img = center_img[i,j]
                        img = img[:, top_row.item() : top_row.item()+crop_size, left_col.item() : left_col.item() + crop_size]
                        # import matplotlib.pyplot as plt
                        # print(img.shape)
                        # print(top_row, left_col)
                        # plt.imshow(img.cpu().permute(1,2,0).numpy())
                        # plt.show()
                        img = img.unsqueeze(dim=0)
                        cano_imgs.append(img)
                cano_imgs = torch.cat(cano_imgs,dim=0)
                cano_imgs = rearrange(cano_imgs, '(b o) n h w -> b (o n) h w', b=bs)
                
                base_agent_pos_pix = agent_pos_pix[:,0:1,:].clone()
                base_agent_pos_pix = base_agent_pos_pix.repeat(1,self.horizon,1)
                cano_label_pix = label_pix - base_agent_pos_pix + crop_size//2
                cano_agent_pos_pix = agent_pos_pix - agent_pos_pix[:,0:1,:]
                cano_agent_pos_pix = -cano_agent_pos_pix + crop_size//2

                nobs['image'] = cano_imgs
                label_pix = cano_label_pix.clamp(min=0,max=95)
                agent_pos_pix = cano_agent_pos_pix

                # print(cano_label_pix.shape)
                # import matplotlib.pyplot as plt
                # for i in range(3):
                #     img = cano_imgs[i,:3,...]
                #     plt.imshow(img.cpu().permute(1,2,0).numpy())
                #     for j in range(16):
                #         act_pix = cano_label_pix[i,j]                
                #         plt.plot(act_pix[1].cpu().numpy(), act_pix[0].cpu().numpy(), marker='x', color=(1-j/16,0,0), markersize=10, markeredgewidth=2)
                #     plt.show()
                
                # vis_cano_image = cano_imgs[:16,:3,...]
                # vis_cano_pix = cano_agent_pos_pix[:16, 0, :]
                # visualize_pusht_images_sequnece(vis_cano_image.cpu(), pos=vis_cano_pix.cpu(), title='transformation', word='sample', transpose=True)

                # vis_cano_image = cano_imgs[:16, 3:,...]
                # vis_cano_pix = cano_agent_pos_pix[:16, 1, :]
                # visualize_pusht_images_sequnece(vis_cano_image.cpu(), pos=vis_cano_pix.cpu(), title='transformation', word='sample', transpose=True)
            
            
            if self.use_pos_map:
                agent_pos_map = pixel2map(agent_pos_pix.reshape(batch_size*self.n_obs_steps, self.action_dim))
                agent_pos_map = agent_pos_map.reshape(batch_size,self.n_obs_steps,imge_shape[-2], imge_shape[-1])
                # print(agent_pos_map.shape)
                # for i in range(10):
                #     print(agent_pos_map[i,0].sum())
                #     print(agent_pos_map[i,0,agent_pos_pix[i,0,0],agent_pos_pix[i,0,1]])
                agent_pos_map = agent_pos_map.unsqueeze(dim=-3)
                # print(agent_pos_map.shape)
                nobs['image'] = nobs['image'].reshape(batch_size, self.n_obs_steps, imge_shape[-3], *imge_shape[-2:])
                nobs['image'] = torch.cat((nobs['image'],agent_pos_map),dim=-3)
                nobs['image'] = nobs['image'].reshape(batch_size, self.n_obs_steps*self.channel, *imge_shape[-2:])
                # print(nobs['image'].shape)
            
                        
            # #-----------------------------
            # print('aug', valid)
            # vis_image = nobs['image'][:16,0:3,...]
            # vis_pix = agent_pos_pix[:16,0,:]
            # visualize_pusht_images_sequnece(vis_image.cpu(), pos=vis_pix.cpu(), title='transformation', word='sample', transpose=True)

            # import matplotlib.pyplot as plt
            # for i in range(3):
            #     img = vis_image[i]
            #     pos_pix = vis_pix[i]
            #     img[:,pos_pix[0]-1:pos_pix[0]+2,pos_pix[1]-1:pos_pix[1]+2]=0
            #     plt.imshow(img.cpu().permute(1,2,0).numpy())
            #     plt.plot(pos_pix[1].cpu().numpy(), pos_pix[0].cpu().numpy(), marker='x', color='red', markersize=10, markeredgewidth=2)
            #     for j in range(16):
            #         act_pix = label_pix[i,j]                
            #         plt.plot(act_pix[1].cpu().numpy(), act_pix[0].cpu().numpy(), marker='x', color=(1-j/16,0,0), markersize=10, markeredgewidth=2)
            #     plt.show()
            # #-----------------------------
            
            pre_action_map, g = self.obs_encoder(obs=nobs)
            pre_action_map_shape = pre_action_map.shape
            pre_action_map = pre_action_map.reshape(-1, pre_action_map_shape[-1]*pre_action_map_shape[-2])

            label = pixel2map(label_pix.reshape(-1, label_pix.shape[-1]))
            loss = F.cross_entropy(input=pre_action_map,target=label)
            
        return loss
