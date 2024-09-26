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
from utils import visualize_pusht_images_sequnece, pos2pixel, pixel2map, pixel2pos
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

        self.obs_encoder = ResUNet(n_input_channel=n_obs_steps*3, n_output_channel=horizon)

        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

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
        print('value', value.shape)
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
            print('image shape',imge_shape)
            visualize_pusht_images_sequnece(nobs['image'][:,0,:,:].cpu())
            visualize_pusht_images_sequnece(nobs['image'][:,1,:,:].cpu())

            nobs['image'] = nobs['image'][:,:self.n_obs_steps,...].reshape(imge_shape[0], self.n_obs_steps*imge_shape[-3], *imge_shape[-2:])
            print(nobs['image'].shape)
            pre_action_map, g = self.obs_encoder(obs=nobs)
            pre_action_map_shape = pre_action_map.shape
            print('action map shape',pre_action_map_shape)
            pre_action_map2 = pre_action_map
        
        # inference action
        pre_action_map = pre_action_map.reshape(-1,pre_action_map_shape[-1]*pre_action_map_shape[-2])
        
        best_action_index = torch.max(pre_action_map,dim=1,keepdim=False)
        print(torch.max(pre_action_map[0]))
        print('max value',best_action_index[0][0])
        best_action_index = best_action_index[1]
        action_pre_x = best_action_index // pre_action_map_shape[-1]
        action_pre_y = best_action_index % pre_action_map_shape[-1]
        action_pred = torch.stack((action_pre_x,action_pre_y),dim=1)
        action_dim = action_pred.shape[-1]
        action_pred = action_pred.reshape(pre_action_map_shape[0], pre_action_map_shape[1], action_dim)
        print('action pred',action_pred.shape)
        visualize_pusht_images_sequnece(pre_action_map2.cpu().unsqueeze(dim=2)[0], action=action_pred.cpu()[0],softmax=False)
        visualize_pusht_images_sequnece(pre_action_map2.cpu().unsqueeze(dim=2)[0], action=action_pred.cpu()[0],softmax=True)

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
            nobs['image'] = nobs['image'][:,:self.n_obs_steps,...].reshape(batch_size, self.n_obs_steps*imge_shape[-3], *imge_shape[-2:])
            # print(nobs['image'].shape,'########################')
            # print(nobs['agent_pos'].shape,'#######################')
            pre_action_map, g = self.obs_encoder(obs=nobs)
            if vis:
                visualize_pusht_images_sequnece(pre_action_map.unsqueeze(dim=2)[0].detach().clone().cpu(), pos=pos_pix.cpu(), action=action_pix.cpu(), title='before normalization')
                print(pre_action_map.shape)
                # for i in range(10):
                #     print(torch.argmax(pre_action_map[0][i]))
            pre_action_map_shape = pre_action_map.shape
            pre_action_map = pre_action_map.reshape(-1,pre_action_map_shape[-1]*pre_action_map_shape[-2])
            

            # inference action
            # best_action_index = torch.max(pre_action_map,dim=1,keepdim=False)[1]
            # print(best_action_index.shape)
            # action_pre_x = best_action_index // pre_action_map_shape[-1]
            # action_pre_y = best_action_index % pre_action_map_shape[-1]
            # action_pred = torch.stack((action_pre_x,action_pre_y),dim=1)
            # print(action_pred[:16,:])
            # print(action_pred.shape)
            # action_dim = action_pred.shape[-1]
            # action_pred = action_pred.reshape(pre_action_map_shape[0],pre_action_map_shape[1],action_dim)
            # print(action_pred[0,:,:])
            # action_pred = pixel2pos(action_pred)

            label_pix = pos2pixel(nactions)
            label = pixel2map(label_pix.reshape(-1, label_pix.shape[-1]))
            loss = F.cross_entropy(input=pre_action_map,target=label)
            
        return loss
