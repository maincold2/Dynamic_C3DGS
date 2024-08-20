#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, update_quaternion
from helper_model import getcolormodel, interpolate_point, interpolate_partuse, interpolate_pointv3, mortonEncode

from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import tinycudann as tcnn
from dahuffman import HuffmanCodec
from dahuffman.huffmancodec import PrefixCodec
import math
from einops import reduce

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp #speical set for visual examples 
        self.scaling_inverse_activation = torch.log #special set for vislual examples

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        self.featureact = torch.sigmoid

        


    def __init__(self, model_args, rgbfuntion="rgbv1"):
        self.active_sh_degree = 0
        self.max_sh_degree = 0#sh_degree  
        self._xyz = torch.empty(0)
        self._mask = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._motion = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._omega = torch.empty(0)
        
        self.rgbdecoder = getcolormodel(rgbfuntion)
    
        self.setup_functions()
        self.delta_t = None
        self.omegamask = None 
        self.maskforems = None 
        self.distancetocamera = None
        self.trbfslinit = None 
        self.ts = None 
        self.trbfoutput = None 
        self.preprocesspoints = False 
        self.addsphpointsscale = 0.8

        
        self.maxz, self.minz =  0.0 , 0.0 
        self.maxy, self.miny =  0.0 , 0.0 
        self.maxx, self.minx =  0.0 , 0.0  
        self.raystart = 0.7
        self.computedtrbfscale = None 
        self.computedopacity = None 
        self.computedscales = None
        
        self.vq_rot = ResidualVQ(dim = 4, codebook_size = model_args.rvq_size_geo, num_quantizers = model_args.rvq_num_geo, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, learnable_codebook=True, in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
        self.vq_sca = ResidualVQ(dim = 3, codebook_size = model_args.rvq_size_geo, num_quantizers = model_args.rvq_num_geo, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, learnable_codebook=True, in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
        
        self.vq_tfea = ResidualVQ(dim = 3, codebook_size = model_args.rvq_size_geo, num_quantizers = model_args.rvq_num_temp, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, learnable_codebook=True, in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
        self.vq_ome = ResidualVQ(dim = 4, codebook_size = model_args.rvq_size_geo, num_quantizers = model_args.rvq_num_temp, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1, ema_update = False, learnable_codebook=True, in_place_codebook_optimizer=lambda *args, **kwargs: torch.optim.Adam(*args, **kwargs, lr=0.0001)).cuda()
        
        self.rvq_bit_geo = math.log2(model_args.rvq_size_geo)
        self.rvq_num_geo = model_args.rvq_num_geo
        self.rvq_bit_temp = math.log2(model_args.rvq_size_temp)
        self.rvq_num_temp = model_args.rvq_num_temp

        self.recolor = tcnn.Encoding(
                 n_input_dims=3,
                 encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": model_args.max_hashmap,
                    "base_resolution": 16,
                    "per_level_scale": 1.447,
                },
            )
        self.mlp_head = tcnn.Network(
                n_input_dims=self.recolor.n_output_dims,
                n_output_dims=6,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    def get_rotation(self, delta_t, rotation, omega):
        self.delta_t = delta_t
        return self.rotation_activation(rotation + delta_t*omega)
    

    @property
    def get_xyz(self):
        return self._xyz
    @property
    def get_trbfcenter(self):
        return self._trbf_center
    @property
    def get_trbfscale(self):
        return self._trbf_scale

    def get_features(self, features_dc, features_t, deltat):
        return torch.cat((features_dc, deltat * features_t), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):

        if self.preprocesspoints == 3:
            pcd = interpolate_point(pcd, 4) 
        
        elif self.preprocesspoints == 4:
            pcd = interpolate_point(pcd, 2) 
        
        elif self.preprocesspoints == 5:
            pcd = interpolate_point(pcd, 6) 

        elif self.preprocesspoints == 6:
            pcd = interpolate_point(pcd, 8) 
        
        elif self.preprocesspoints == 7:
            pcd = interpolate_point(pcd, 16) 
        elif self.preprocesspoints == 8:
            pcd = interpolate_pointv3(pcd, 4) 
        elif self.preprocesspoints == 14:
            pcd = interpolate_partuse(pcd, 2) 
        
        elif self.preprocesspoints == 15:
            pcd = interpolate_partuse(pcd, 4) 

        elif self.preprocesspoints == 16:
            pcd = interpolate_partuse(pcd, 8) 
        
        elif self.preprocesspoints == 17:
            pcd = interpolate_partuse(pcd, 16) 
        else:
            pass 
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        times = torch.tensor(np.asarray(pcd.times)).float().cuda()


        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        features9channel = torch.cat((fused_color, fused_color), dim=1)
        
        N, _ = fused_color.shape

        fomega = torch.zeros((N, 3), dtype=torch.float, device="cuda")
        self._features_t =  nn.Parameter(fomega.contiguous().requires_grad_(True))
        

        self._mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        omega = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        self._omega = nn.Parameter(omega.requires_grad_(True))
        
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        motion = torch.zeros((fused_point_cloud.shape[0], 9), device="cuda")# x1, x2, x3,  y1,y2,y3, z1,z2,z3
        self._motion = nn.Parameter(motion.requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        


        self._trbf_center = nn.Parameter(times.contiguous().requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.ones((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True)) 

        ## store gradients


        if self.trbfslinit is not None:
            nn.init.constant_(self._trbf_scale, self.trbfslinit) # too large ?
        else:
            nn.init.constant_(self._trbf_scale, 0) # too large ?

        nn.init.constant_(self._features_t, 0)
        nn.init.constant_(self._omega, 0)
        self.rgb_grd = {}



        self.maxz, self.minz = torch.amax(self._xyz[:,2]), torch.amin(self._xyz[:,2]) 
        self.maxy, self.miny = torch.amax(self._xyz[:,1]), torch.amin(self._xyz[:,1]) 
        self.maxx, self.minx = torch.amax(self._xyz[:,0]), torch.amin(self._xyz[:,0]) 
        self.maxz = min((self.maxz, 200.0)) # some outliers in the n4d datasets.. 
        
       
        

        for name, W in self.rgbdecoder.named_parameters():
            if 'weight' in name:
                self.rgb_grd[name] = torch.zeros_like(W, requires_grad=False).cuda() #self.rgb_grd[name] + W.grad.clone()
            elif 'bias' in name:
                print('not implemented')
                quit()
    def cache_gradient(self):
        self._xyz_grd += self._xyz.grad.clone()
        self._features_dc_grd += self.recolor.params.grad.clone()
        self._features_t_grd += self._features_t.grad.clone() # self._features_t_grd
        self._scaling_grd += self._scaling.grad.clone()
        self._rotation_grd += self._rotation.grad.clone()
        self._opacity_grd += self._opacity.grad.clone()
        self._trbf_center_grd += self._trbf_center.grad.clone()
        self._trbf_scale_grd += self._trbf_scale.grad.clone()
        self._motion_grd += self._motion.grad.clone()
        self._omega_grd += self._omega.grad.clone()
        self._mask_grd += self._mask.grad.clone()

        for name, W in self.rgbdecoder.named_parameters():
            if 'weight' in name:
                self.rgb_grd[name] = self.rgb_grd[name] + W.grad.clone()
    def zero_gradient_cache(self):

        self._xyz_grd = torch.zeros_like(self._xyz, requires_grad=False)
        self._features_dc_grd = torch.zeros_like(self.recolor.params, requires_grad=False)
        self._features_t_grd = torch.zeros_like(self._features_t, requires_grad=False)


        self._scaling_grd = torch.zeros_like(self._scaling, requires_grad=False)
        self._rotation_grd = torch.zeros_like(self._rotation, requires_grad=False)
        self._opacity_grd = torch.zeros_like(self._opacity, requires_grad=False)
        self._trbf_center_grd = torch.zeros_like(self._trbf_center, requires_grad=False)
        self._trbf_scale_grd = torch.zeros_like(self._trbf_scale, requires_grad=False)
        self._motion_grd = torch.zeros_like(self._motion, requires_grad=False)
        self._omega_grd = torch.zeros_like(self._omega, requires_grad=False)
        self._mask_grd = torch.zeros_like(self._mask, requires_grad=False)



        for name in self.rgb_grd.keys():
            self.rgb_grd[name].zero_()

    def set_batch_gradient(self, cnt):
        ratio = 1/cnt
        self.recolor.params.grad = self._features_dc_grd * ratio
        self._features_t.grad = self._features_t_grd * ratio 
        self._xyz.grad = self._xyz_grd * ratio
        self._scaling.grad = self._scaling_grd * ratio
        self._rotation.grad = self._rotation_grd * ratio
        self._opacity.grad = self._opacity_grd * ratio
        self._trbf_center.grad = self._trbf_center_grd * ratio
        self._trbf_scale.grad = self._trbf_scale_grd* ratio
        self._motion.grad = self._motion_grd * ratio
        self._omega.grad = self._omega_grd * ratio
        self._mask.grad = self._mask_grd * ratio

        for name, W in self.rgbdecoder.named_parameters():
            if 'weight' in name:
                W.grad = self.rgb_grd[name] * ratio


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.rgbdecoder.cuda()
         # self._features_t
        
        other_params = []
        for params in self.recolor.parameters():
            other_params.append(params)
        for params in self.mlp_head.parameters():
            other_params.append(params)
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_t], 'lr': training_args.featuret_lr, "name": "f_t"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._omega], 'lr': training_args.omega_lr, "name": "omega"},
            {'params': [self._trbf_center], 'lr': training_args.trbfc_lr, "name": "trbf_center"},
            {'params': [self._trbf_scale], 'lr': training_args.trbfs_lr, "name": "trbf_scale"},
            {'params': [self._motion], 'lr':  training_args.position_lr_init * self.spatial_lr_scale * 0.5 * training_args.movelr , "name": "motion"},
            
            {'params': [self._mask], 'lr':  training_args.mask_lr , "name": "mask"},
            
            {'params': list(self.rgbdecoder.parameters()), 'lr': training_args.rgb_lr, "name": "decoder"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.optimizer_net = torch.optim.Adam(other_params, lr=training_args.net_lr, eps=1e-15)
        self.scheduler_net = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
            self.optimizer_net, start_factor=0.01, total_iters=100
        ),
            torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_net,
            milestones=training_args.net_lr_step,
            gamma=0.33,
        ),
        ]
        )
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        print("move decoder to cuda")
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
    
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z','trbf_center', 'trbf_scale' ,'nx', 'ny', 'nz'] # 'trbf_center', 'trbf_scale' 
        # All channels except the 3 DC
        for i in range(self._motion.shape[1]):
            l.append('motion_{}'.format(i))
        for i in range(self._features_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._omega.shape[1]):
            l.append('omega_{}'.format(i))
        


        for i in range(self._features_t.shape[1]):
            l.append('f_t_{}'.format(i))
        
        return l

    
    def save_npz(self, path):
        mkdir_p(os.path.dirname(path))

        save_dict = dict()
        
        save_dict["xyz"] = self._xyz.detach().cpu().half().numpy()
        save_dict["motion"] = self._motion.detach().cpu().half().numpy()
        save_dict["opacity"] = self._opacity.detach().cpu().half().numpy()
        save_dict["tcen"] = self._trbf_center.detach().cpu().half().numpy()
        save_dict["tsca"] = self._trbf_scale.detach().cpu().half().numpy()
        save_dict["scale"] = np.packbits(np.unpackbits(self.sca_idx.unsqueeze(-1).cpu().numpy().astype(np.uint8), axis=-1, count=8, bitorder='little').flatten(), axis=None)
        save_dict["rotation"] = np.packbits(np.unpackbits(self.rot_idx.unsqueeze(-1).cpu().numpy().astype(np.uint8), axis=-1, count=8, bitorder='little').flatten(), axis=None)
        save_dict["omega"] = np.packbits(np.unpackbits(self.ome_idx.unsqueeze(-1).cpu().numpy().astype(np.uint8), axis=-1, count=8, bitorder='little').flatten(), axis=None)
        save_dict["tfea"] = np.packbits(np.unpackbits(self.tfea_idx.unsqueeze(-1).cpu().numpy().astype(np.uint8), axis=-1, count=8, bitorder='little').flatten(), axis=None)
        save_dict["hash"] = self.recolor.params.cpu().half().numpy()
        save_dict["mlp"] = self.mlp_head.params.cpu().half().numpy()
        save_dict["codebook_scale"] = self.vq_sca.cpu().state_dict()
        save_dict["codebook_rotation"] = self.vq_rot.cpu().state_dict()
        save_dict["codebook_omega"] = self.vq_ome.cpu().state_dict()
        save_dict["codebook_tfea"] = self.vq_tfea.cpu().state_dict()
        save_dict["rvq_info_geo"] = np.array([self.rvq_num_geo, self.rvq_bit_geo])
        save_dict["rvq_info_temp"] = np.array([self.rvq_num_temp, self.rvq_bit_temp])
        save_dict["rgb_dec"] = self.rgbdecoder.cpu().state_dict()
        
        np.savez(path, **save_dict)
        
    def save_npz_pp(self, path):
        mkdir_p(os.path.dirname(path))

        save_dict = dict()
        
        save_dict["xyz"] = self._xyz.detach().cpu().half().numpy()
        save_dict["motion"] = self._motion.detach().cpu().half().numpy()
        save_dict["opacity"] = np.frombuffer(self.huf_opa, dtype=np.uint8)
        save_dict["tcen"] = np.frombuffer(self.huf_tcen, dtype=np.uint8)
        save_dict["tsca"] = np.frombuffer(self.huf_tsca, dtype=np.uint8)
        save_dict["scale"] = np.frombuffer(self.huf_sca, dtype=np.uint8)
        save_dict["rotation"] = np.frombuffer(self.huf_rot, dtype=np.uint8)
        save_dict["omega"] = np.frombuffer(self.huf_ome, dtype=np.uint8)
        save_dict["tfea"] = np.frombuffer(self.huf_tfea, dtype=np.uint8)
        save_dict["hash"] = np.frombuffer(self.huf_hash, dtype=np.uint8)
        save_dict["mlp"] = self.mlp_head.params.cpu().half().numpy()
        save_dict["huftable_opacity"] = self.tab_opa
        save_dict["huftable_tcen"] = self.tab_tcen
        save_dict["huftable_tsca"] = self.tab_tsca
        save_dict["huftable_scale"] = self.tab_sca
        save_dict["huftable_rotation"] = self.tab_rot
        save_dict["huftable_omega"] = self.tab_ome
        save_dict["huftable_tfea"] = self.tab_tfea
        save_dict["huftable_hash"] = self.tab_hash
        save_dict["codebook_scale"] = self.vq_sca.cpu().state_dict()
        save_dict["codebook_rotation"] = self.vq_rot.cpu().state_dict()
        save_dict["codebook_omega"] = self.vq_ome.cpu().state_dict()
        save_dict["codebook_tfea"] = self.vq_tfea.cpu().state_dict()
        save_dict["minmax_opacity"] = self.minmax_opa.numpy()
        save_dict["minmax_tcen"] = self.minmax_tcen.numpy()
        save_dict["minmax_tsca"] = self.minmax_tsca.numpy()
        save_dict["minmax_hash"] = self.minmax_hash.numpy()
        save_dict["rvq_info_geo"] = np.array([self.rvq_num_geo, self.rvq_bit_geo])
        save_dict["rvq_info_temp"] = np.array([self.rvq_num_temp, self.rvq_bit_temp])
        save_dict["rgb_dec"] = self.rgbdecoder.cpu().state_dict()
        
        np.savez_compressed(path+"_pp", **save_dict)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        self.precompute()
        f_dc = self._features_dc.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()


        trbf_center= self._trbf_center.detach().cpu().numpy()

        trbf_scale = self._trbf_scale.detach().cpu().numpy()
        motion = self._motion.detach().cpu().numpy()

        omega = self._omega.detach().cpu().numpy()

        f_t =  self._features_t.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, trbf_center, trbf_scale, normals, motion, f_dc, opacities, scale, rotation, omega, f_t), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        model_fname = path.replace(".ply", ".pt")
        print(f'Saving model checkpoint to: {model_fname}')
        torch.save(self.rgbdecoder.state_dict(), model_fname)

    def post_quant(self, param, prune=False):
        max_val = torch.amax(param)
        min_val = torch.amin(param)
        if prune:
            param = param*(torch.abs(param) > 0.1)
        param = (param - min_val)/(max_val - min_val)
        quant = torch.round(param * 255.0)
        out = (max_val - min_val)*quant/255.0 + min_val
        return torch.nn.Parameter(out), quant, torch.tensor([min_val, max_val])
    
    def huffman_encode(self, param):
        input_code_list = param.view(-1).tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        codec = HuffmanCodec.from_data(input_code_list)

        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        total_mb = total_bits/8/10**6
        
        return total_mb, codec.encode(input_code_list), codec.get_code_table()
    
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def zero_omega(self, threhold=0.15):
        scales = self.get_scaling
        omegamask = torch.sum(torch.abs(self._omega), dim=1) > threhold # default 
        scalemask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scalemaskb = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        pointopacity = self.get_opacity
        opacitymask = pointopacity > 0.7

        mask = torch.logical_and(torch.logical_and(omegamask.unsqueeze(1), scalemask), torch.logical_and(scalemaskb, opacitymask))
        omeganew = mask.float() * self._omega
        optimizable_tensors = self.replace_tensor_to_optimizer(omeganew, "omega")
        self._omega = optimizable_tensors["omega"]
        return mask
    def zero_omegabymotion(self, threhold=0.15):
        scales = self.get_scaling
        omegamask = torch.sum(torch.abs(self._motion[:, 0:3]), dim=1) > 0.3 #  #torch.sum(torch.abs(self._omega), dim=1) > threhold # default 
        scalemask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scalemaskb = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        pointopacity = self.get_opacity
        opacitymask = pointopacity > 0.7

        

        mask = torch.logical_and(torch.logical_and(omegamask.unsqueeze(1), scalemask), torch.logical_and(scalemaskb, opacitymask))
        
        
        omeganew = mask.float() * self._omega
        optimizable_tensors = self.replace_tensor_to_optimizer(omeganew, "omega")
        self._omega = optimizable_tensors["omega"]
        return mask


    def zero_omegav2(self, threhold=0.15):
        scales = self.get_scaling
        omegamask = torch.sum(torch.abs(self._omega), dim=1) > threhold # default 
        scalemask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scalemaskb = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        pointopacity = self.get_opacity
        opacitymask = pointopacity > 0.7

        mask = torch.logical_and(torch.logical_and(omegamask.unsqueeze(1), scalemask), torch.logical_and(scalemaskb, opacitymask))
        omeganew = mask.float() * self._omega
        rotationew = self.get_rotation(self.delta_t)


        optimizable_tensors = self.replace_tensor_to_optimizer(omeganew, "omega")
        self._omega = optimizable_tensors["omega"]


        optimizable_tensors = self.replace_tensor_to_optimizer(rotationew, "rotation")
        self._rotation = optimizable_tensors["rotation"]
        return mask
    
    def load_model(self, path):
        if os.path.isfile(path + '_pp.npz'):
            path = path + '_pp.npz'
            print("Loading ", path)
            load_dict = np.load(path, allow_pickle=True)

            codec = PrefixCodec(load_dict["huftable_opacity"].item())
            opacity = torch.tensor(codec.decode(load_dict["opacity"]))
            
            codec = PrefixCodec(load_dict["huftable_tcen"].item())
            t_center = torch.tensor(codec.decode(load_dict["tcen"]))
            
            codec = PrefixCodec(load_dict["huftable_tsca"].item())
            t_scale = torch.tensor(codec.decode(load_dict["tsca"]))

            codec = PrefixCodec(load_dict["huftable_scale"].item())
            scale = codec.decode(load_dict["scale"])

            codec = PrefixCodec(load_dict["huftable_rotation"].item())
            rotation = codec.decode(load_dict["rotation"])

            codec = PrefixCodec(load_dict["huftable_omega"].item())
            omega = codec.decode(load_dict["omega"])

            codec = PrefixCodec(load_dict["huftable_tfea"].item())
            t_feature = codec.decode(load_dict["tfea"])
            
            codec = PrefixCodec(load_dict["huftable_hash"].item())
            hashgrid = torch.tensor(codec.decode(load_dict["hash"]))

            opacity = (float(load_dict["minmax_opacity"][1]) - float(load_dict["minmax_opacity"][0]))*opacity/255.0 + float(load_dict["minmax_opacity"][0])
            t_center = (float(load_dict["minmax_tcen"][1]) - float(load_dict["minmax_tcen"][0]))*t_center/255.0 + float(load_dict["minmax_tcen"][0])
            t_scale = (float(load_dict["minmax_tsca"][1]) - float(load_dict["minmax_tsca"][0]))*t_scale/255.0 + float(load_dict["minmax_tsca"][0])
            hashgrid = (float(load_dict["minmax_hash"][1]) - float(load_dict["minmax_hash"][0]))*hashgrid/255.0 + float(load_dict["minmax_hash"][0])

            self.vq_sca.load_state_dict(load_dict["codebook_scale"].item())
            self.vq_rot.load_state_dict(load_dict["codebook_rotation"].item())
            self.vq_ome.load_state_dict(load_dict["codebook_omega"].item())
            self.vq_tfea.load_state_dict(load_dict["codebook_tfea"].item())
            
            rvq_num_geo = int(load_dict["rvq_info_geo"][0])
            rvq_bit_geo = int(load_dict["rvq_info_geo"][1])
            rvq_num_temp = int(load_dict["rvq_info_temp"][0])
            rvq_bit_temp = int(load_dict["rvq_info_temp"][1])

            scale_codes = self.vq_sca.get_codes_from_indices(torch.tensor(scale).cuda().reshape(-1,1,rvq_num_geo))
            scale = self.vq_sca.project_out(reduce(scale_codes, 'q ... -> ...', 'sum'))
            rotation_codes = self.vq_rot.get_codes_from_indices(torch.tensor(rotation).cuda().reshape(-1,1,rvq_num_geo))
            rotation = self.vq_rot.project_out(reduce(rotation_codes, 'q ... -> ...', 'sum'))
            
            omega_codes = self.vq_ome.get_codes_from_indices(torch.tensor(omega).cuda().reshape(-1,1,rvq_num_temp))
            omega = self.vq_ome.project_out(reduce(omega_codes, 'q ... -> ...', 'sum'))
            tfea_codes = self.vq_tfea.get_codes_from_indices(torch.tensor(t_feature).cuda().reshape(-1,1,rvq_num_temp))
            t_feature = self.vq_tfea.project_out(reduce(tfea_codes, 'q ... -> ...', 'sum'))

            self._xyz = nn.Parameter(torch.from_numpy(load_dict["xyz"]).cuda().float().requires_grad_(True))
            self._motion = nn.Parameter(torch.from_numpy(load_dict["motion"]).cuda().float().requires_grad_(True))
            self._opacity = nn.Parameter(opacity.cuda().reshape(-1,1).float().requires_grad_(True))
            self._trbf_center = nn.Parameter(t_center.cuda().reshape(-1,1).float().requires_grad_(True))
            self._trbf_scale = nn.Parameter(t_scale.cuda().reshape(-1,1).float().requires_grad_(True))
            self._scaling = nn.Parameter(scale.squeeze(1).requires_grad_(True))
            self._rotation = nn.Parameter(rotation.squeeze(1).requires_grad_(True))
            self._omega = nn.Parameter(omega.squeeze(1).requires_grad_(True))
            self._features_t = nn.Parameter(t_feature.squeeze(1).requires_grad_(True))
            self.recolor.params = nn.Parameter(hashgrid.cuda().half().requires_grad_(True))
            self.mlp_head.params = nn.Parameter(torch.from_numpy(load_dict["mlp"]).cuda().half().requires_grad_(True))
            self.rgbdecoder.load_state_dict(load_dict["rgb_dec"].item())
            self.precompute()
            
        elif os.path.isfile(path + '.npz'):
            path = path + '.npz'
            print("Loading ", path)
            load_dict = np.load(path, allow_pickle=True)

            rvq_num_geo = int(load_dict["rvq_info_geo"][0])
            rvq_bit_geo = int(load_dict["rvq_info_geo"][1])
            rvq_num_temp = int(load_dict["rvq_info_temp"][0])
            rvq_bit_temp = int(load_dict["rvq_info_temp"][1])
            
            scale = np.packbits(np.unpackbits(load_dict["scale"], axis=None)[:load_dict["xyz"].shape[0]*rvq_num_geo*rvq_bit_geo].reshape(-1, rvq_bit_geo), axis=-1, bitorder='little')
            rotation = np.packbits(np.unpackbits(load_dict["rotation"], axis=None)[:load_dict["xyz"].shape[0]*rvq_num_geo*rvq_bit_geo].reshape(-1, rvq_bit_geo), axis=-1, bitorder='little')
            
            omega = np.packbits(np.unpackbits(load_dict["omega"], axis=None)[:load_dict["xyz"].shape[0]*rvq_num_temp*rvq_bit_temp].reshape(-1, rvq_bit_temp), axis=-1, bitorder='little')
            t_feature = np.packbits(np.unpackbits(load_dict["tfea"], axis=None)[:load_dict["xyz"].shape[0]*rvq_num_temp*rvq_bit_temp].reshape(-1, rvq_bit_temp), axis=-1, bitorder='little')

            self.vq_sca.load_state_dict(load_dict["codebook_scale"].item())
            self.vq_rot.load_state_dict(load_dict["codebook_rotation"].item())
            self.vq_ome.load_state_dict(load_dict["codebook_omega"].item())
            self.vq_tfea.load_state_dict(load_dict["codebook_tfea"].item())
            scale_codes = self.vq_sca.get_codes_from_indices(torch.from_numpy(scale).cuda().reshape(-1,1,rvq_num_geo).long())
            scale = self.vq_sca.project_out(reduce(scale_codes, 'q ... -> ...', 'sum'))
            rotation_codes = self.vq_rot.get_codes_from_indices(torch.from_numpy(rotation).cuda().reshape(-1,1,rvq_num_geo).long())
            rotation = self.vq_rot.project_out(reduce(rotation_codes, 'q ... -> ...', 'sum'))
            
            omega_codes = self.vq_ome.get_codes_from_indices(torch.from_numpy(omega).cuda().reshape(-1,1,rvq_num_temp).long())
            omega = self.vq_ome.project_out(reduce(omega_codes, 'q ... -> ...', 'sum'))
            tfea_codes = self.vq_tfea.get_codes_from_indices(torch.from_numpy(t_feature).cuda().reshape(-1,1,rvq_num_temp).long())
            t_feature = self.vq_tfea.project_out(reduce(tfea_codes, 'q ... -> ...', 'sum'))

            self._xyz = nn.Parameter(torch.from_numpy(load_dict["xyz"]).cuda().float().requires_grad_(True))
            self._motion = nn.Parameter(torch.from_numpy(load_dict["motion"]).cuda().float().requires_grad_(True))
            self._opacity = nn.Parameter(torch.from_numpy(load_dict["opacity"]).reshape(-1,1).cuda().float().requires_grad_(True))
            self._trbf_center = nn.Parameter(torch.from_numpy(load_dict["tcen"]).reshape(-1,1).cuda().float().requires_grad_(True))
            self._trbf_scale = nn.Parameter(torch.from_numpy(load_dict["tsca"]).reshape(-1,1).cuda().float().requires_grad_(True))
            self._scaling = nn.Parameter(scale.squeeze(1).requires_grad_(True))
            self._rotation = nn.Parameter(rotation.squeeze(1).requires_grad_(True))
            self._omega = nn.Parameter(omega.squeeze(1).requires_grad_(True))
            self._features_t = nn.Parameter(t_feature.squeeze(1).requires_grad_(True))
            self.recolor.params = nn.Parameter(torch.from_numpy(load_dict["hash"]).cuda().half().requires_grad_(True))
            self.mlp_head.params = nn.Parameter(torch.from_numpy(load_dict["mlp"]).cuda().half().requires_grad_(True))
            self.rgbdecoder.load_state_dict(load_dict["rgb_dec"].item())
            self.precompute()
        else:
            self.load_ply(path+".ply")
    
    def load_ply(self, path):
        plydata = PlyData.read(path)
        ckpt = torch.load(path.replace(".ply", ".pt"))
        self.rgbdecoder.load_state_dict(ckpt)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        trbf_center= np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale = np.asarray(plydata.elements[0]["trbf_scale"])[..., np.newaxis]

        motion_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion")]
        nummotion = 9
        motion = np.zeros((xyz.shape[0], nummotion))
        for i in range(nummotion):
            motion[:, i] = np.asarray(plydata.elements[0]["motion_"+str(i)])


        dc_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc")]
        num_dc_features = len(dc_f_names)

        features_dc = np.zeros((xyz.shape[0], num_dc_features))
        for i in range(num_dc_features):
            features_dc[:, i] = np.asarray(plydata.elements[0]["f_dc_"+str(i)])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        #assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])


        ft_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_t")]
        ftomegas = np.zeros((xyz.shape[0], len(ft_names)))
        for idx, attr_name in enumerate(ft_names):
            ftomegas[:, idx] = np.asarray(plydata.elements[0][attr_name])



        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_center = nn.Parameter(torch.tensor(trbf_center, dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.tensor(trbf_scale, dtype=torch.float, device="cuda").requires_grad_(True))
        self._motion = nn.Parameter(torch.tensor(motion, dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_t = nn.Parameter(torch.tensor(ftomegas, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        
        self.computedopacity =self.opacity_activation(self._opacity)
        self.computedscales = torch.exp(self._scaling) # change not very large
        self.computedtrbfscale = torch.exp(self._trbf_scale) 


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) == 1 and group["name"] != 'decoder' :
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        self._features_t = optimizable_tensors["f_t"]
        self._mask = optimizable_tensors["mask"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.omegamask is not None :
            self.omegamask = self.omegamask[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) == 1 and group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_opacities, new_scaling, new_rotation, new_trbf_center, new_trbfscale, new_motion, new_omega, new_featuret, new_mask):
        d = {"xyz": new_xyz,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "trbf_center" : new_trbf_center,
        "trbf_scale" : new_trbfscale,
        "motion": new_motion,
        "omega": new_omega,
        "f_t": new_featuret,
        "mask": new_mask}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_t = optimizable_tensors["f_t"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        self._mask = optimizable_tensors["mask"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    

    def densify_and_splitv2(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_trbf_center = self._trbf_center[selected_pts_mask].repeat(N,1)
        new_trbf_center = torch.rand_like(new_trbf_center) #* 0.5
        new_trbf_scale = self._trbf_scale[selected_pts_mask].repeat(N,1)
        new_motion = self._motion[selected_pts_mask].repeat(N,1)
        new_omega = self._omega[selected_pts_mask].repeat(N,1)
        new_features_t = self._features_t[selected_pts_mask].repeat(N,1)
        new_mask = self._mask[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_opacity, new_scaling, new_rotation, new_trbf_center, new_trbf_scale, new_motion, new_omega, new_features_t, new_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        
        new_xyz = self._xyz[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        #new_trbf_center =  torch.rand((self._trbf_center[selected_pts_mask].shape[0], 1), device="cuda")  #self._trbf_center[selected_pts_mask]
        #new_trbf_center =  self._trbf_center[selected_pts_mask] # 
        new_trbf_center =  torch.rand((self._trbf_center[selected_pts_mask].shape[0], 1), device="cuda")  #self._trbf_center[selected_pts_mask]
        new_trbfscale = self._trbf_scale[selected_pts_mask]
        new_motion = self._motion[selected_pts_mask]
        new_omega = self._omega[selected_pts_mask]
        new_featuret = self._features_t[selected_pts_mask]
        new_mask = self._mask[selected_pts_mask]
        N, c= new_featuret.shape
        #self.trbfoutput = torch.cat((self.trbfoutput, torch.zeros(N , 1).to(self.trbfoutput)))
        self.densification_postfix(new_xyz, new_opacities, new_scaling, new_rotation, new_trbf_center, new_trbfscale, new_motion, new_omega, new_featuret, new_mask)


    def densify_pruneclone(self, max_grad, min_opacity, extent, max_screen_size, splitN=1):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_splitv2(grads, max_grad, extent, 2)

        prune_mask = (torch.sigmoid(self._mask) <= 0.01).squeeze()
        self.prune_points(prune_mask)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size  
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        torch.cuda.empty_cache()

    def precompute(self):
        xyz = self.contract_to_unisphere(self.get_xyz, torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
        self._features_dc = self.recolor(xyz)
        self._features_dc = self.mlp_head(self.recolor(xyz))
        torch.cuda.empty_cache()
        
    def mask_prune(self):
        prune_mask = (torch.sigmoid(self._mask) <= 0.01).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
    
    def final_prune(self, compress=False):
        self.mask_prune()
        
        if compress:
            self._sort_morton()
            
        for m in self.vq_sca.layers:
            m.training = False
        for m in self.vq_rot.layers: 
            m.training = False
        for m in self.vq_tfea.layers:
            m.training = False
        for m in self.vq_ome.layers: 
            m.training = False
        
        self._xyz = self._xyz.clone().half().float()
        self._scaling, self.sca_idx, _ = self.vq_sca(self._scaling.unsqueeze(1))
        self._rotation, self.rot_idx, _ = self.vq_rot(self._rotation.unsqueeze(1))
        self._scaling = self._scaling.squeeze()
        self._rotation = self._rotation.squeeze()
        
        self._features_t, self.tfea_idx, _ = self.vq_tfea(self._features_t.unsqueeze(1))
        self._omega, self.ome_idx, _ = self.vq_ome(self._omega.unsqueeze(1))
        self._omega = self._omega.squeeze()
        self._features_t = self._features_t.squeeze()
        
        num_G = self._xyz.shape[0]
        position_mb = num_G*3*16/8/10**6
        motion_mb = num_G*self._motion.shape[1]*16/8/10**6

        scale_mb = num_G*self.rvq_bit_geo*self.rvq_num_geo/8/10**6 + 2**self.rvq_bit_geo*self.rvq_num_geo*3*32/8/10**6
        rotation_mb = num_G*self.rvq_bit_geo*self.rvq_num_geo/8/10**6 + 2**self.rvq_bit_geo*self.rvq_num_geo*4*32/8/10**6
        
        rvq_num = 4
        omega_mb = num_G*self.rvq_bit_temp*self.rvq_num_temp/8/10**6 + 2**self.rvq_bit_temp*self.rvq_num_temp*3*32/8/10**6
        tfea_mb = num_G*self.rvq_bit_temp*self.rvq_num_temp/8/10**6 + 2**self.rvq_bit_temp*self.rvq_num_temp*4*32/8/10**6
        
        opacity_mb = num_G*16/8/10**6
        tcen_mb = num_G*16/8/10**6
        tsca_mb = num_G*16/8/10**6
        
        hash_mb = self.recolor.params.shape[0]*16/8/10**6
        mlp_mb = self.mlp_head.params.shape[0]*16/8/10**6
        sum_mb = position_mb+motion_mb+scale_mb+rotation_mb+omega_mb+tfea_mb+opacity_mb+tcen_mb+tsca_mb+hash_mb+mlp_mb
        
        mb_str = "Storage\nposition: "+str(position_mb)+"\nmotion: "+str(motion_mb)+"\nscale: "+str(scale_mb)+"\nrotation: "+str(rotation_mb)+"\nomega: "+str(omega_mb)+"\ntime_feature: "+str(tfea_mb)+"\nopacity: "+str(opacity_mb)+"\ntime_center: "+str(tcen_mb)+"\ntime_scale: "+str(tsca_mb)+"\nhash: "+str(hash_mb)+"\nmlp: "+str(mlp_mb)+"\ntotal: "+str(sum_mb)+" MB"
        
        if compress:
            _, self.quant_opa, self.minmax_opa = self.post_quant(self.get_opacity)
            _, self.quant_tcen, self.minmax_tcen = self.post_quant(self._trbf_center)
            _, self.quant_tsca, self.minmax_tsca = self.post_quant(self._trbf_scale)
            _, self.quant_hash, self.minmax_hash = self.post_quant(self.recolor.params)#, True)
 
            scale_mb, self.huf_sca, self.tab_sca = self.huffman_encode(self.sca_idx) 
            scale_mb += 2**self.rvq_bit_geo*self.rvq_num_geo*3*32/8/10**6
            rotation_mb, self.huf_rot, self.tab_rot = self.huffman_encode(self.rot_idx)
            rotation_mb += 2**self.rvq_bit_geo*self.rvq_num_geo*4*32/8/10**6
            
            omega_mb, self.huf_ome, self.tab_ome = self.huffman_encode(self.ome_idx) 
            omega_mb += 2**self.rvq_bit_temp*self.rvq_num_temp*3*32/8/10**6
            tfea_mb, self.huf_tfea, self.tab_tfea = self.huffman_encode(self.tfea_idx)
            tfea_mb += 2**self.rvq_bit_temp*self.rvq_num_temp*4*32/8/10**6
            
            opacity_mb, self.huf_opa, self.tab_opa = self.huffman_encode(self.quant_opa)
            tcen_mb, self.huf_tcen, self.tab_tcen = self.huffman_encode(self.quant_tcen)
            tsca_mb, self.huf_tsca, self.tab_tsca = self.huffman_encode(self.quant_tsca)
            hash_mb, self.huf_hash, self.tab_hash = self.huffman_encode(self.quant_hash)
            mlp_mb = self.mlp_head.params.shape[0]*16/8/10**6
            sum_mb = position_mb+motion_mb+scale_mb+rotation_mb+omega_mb+tfea_mb+opacity_mb+tcen_mb+tsca_mb+hash_mb+mlp_mb
            
            mb_str = mb_str+"\n\nAfter PP\nposition: "+str(position_mb)+"\nmotion: "+str(motion_mb)+"\nscale: "+str(scale_mb)+"\nrotation: "+str(rotation_mb)+"\nomega: "+str(omega_mb)+"\ntime_feature: "+str(tfea_mb)+"\nopacity: "+str(opacity_mb)+"\ntime_center: "+str(tcen_mb)+"\ntime_scale: "+str(tsca_mb)+"\nhash: "+str(hash_mb)+"\nmlp: "+str(mlp_mb)+"\ntotal: "+str(sum_mb)+" MB"

        self._opacity = self.get_opacity.clone().half().float()
        self._trbf_center = self._trbf_center.clone().half().float()
        self._trbf_scale = self._trbf_scale.clone().half().float()
        torch.cuda.empty_cache()
        return mb_str
    
        
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1



    def addgaussians(self, baduvidx, viewpoint_cam, depthmap, gt_image, numperay=3, ratioend=2, trbfcenter=0.5,depthmax=None,shuffle=False):
        def pix2ndc(v, S):
            return (v * 2.0 + 1.0) / S - 1.0
        ratiaolist = torch.linspace(self.raystart, ratioend, numperay) # 0.7 to ratiostart
        rgbs = gt_image[:, baduvidx[:,0], baduvidx[:,1]]
        rgbs = rgbs.permute(1,0)
        featuredc = torch.cat((rgbs, torch.zeros_like(rgbs)), dim=1)# should we add the feature dc with non zero values?

        depths = depthmap[:, baduvidx[:,0], baduvidx[:,1]]
        depths = depths.permute(1,0) # only use depth map > 15 .

        depths = torch.ones_like(depths) * depthmax # use the max local depth for the scene ?

        
        u = baduvidx[:,0] # hight y
        v = baduvidx[:,1] # weidth  x 
        Npoints = u.shape[0]
          
        new_xyz = []
        new_scaling = []
        new_rotation = []
        new_opacity = []
        new_trbf_center = []
        new_trbf_scale = []
        new_motion = []
        new_omega = []
        new_featuret = [ ]

        camera2wold = viewpoint_cam.world_view_transform.T.inverse()
        projectinverse = viewpoint_cam.projection_matrix.T.inverse()
        maxz, minz = self.maxz, self.minz 
        maxy, miny = self.maxy, self.miny 
        maxx, minx = self.maxx, self.minx  
        

        for zscale in ratiaolist :
            ndcu, ndcv = pix2ndc(u, viewpoint_cam.image_height), pix2ndc(v, viewpoint_cam.image_width)
            # targetPz = depths*zscale # depth in local cameras..
            if shuffle == True:
                randomdepth = torch.rand_like(depths) - 0.5 # -0.5 to 0.5
                targetPz = (depths + depths/10*(randomdepth)) *zscale 
            else:
                targetPz = depths*zscale # depth in local cameras..
            
            ndcu = ndcu.unsqueeze(1)
            ndcv = ndcv.unsqueeze(1)


            ndccamera = torch.cat((ndcv, ndcu,   torch.ones_like(ndcu) * (1.0) , torch.ones_like(ndcu)), 1) # N,4 ...
            
            localpointuv = ndccamera @ projectinverse.T 

            diretioninlocal = localpointuv / localpointuv[:,3:] # ray direction in camera space 


            rate = targetPz / diretioninlocal[:, 2:3] #  
            
            localpoint = diretioninlocal * rate

            localpoint[:, -1] = 1
            
            
            worldpointH = localpoint @ camera2wold.T  #myproduct4x4batch(localpoint, camera2wold) # 
            worldpoint = worldpointH / worldpointH[:, 3:] #  

            xyz = worldpoint[:, :3] 
            distancetocameracenter = viewpoint_cam.camera_center - xyz
            distancetocameracenter = torch.norm(distancetocameracenter, dim=1)

            xmask = torch.logical_and(xyz[:, 0] > minx, xyz[:, 0] < maxx )
            selectedmask = torch.logical_or(xmask, torch.logical_not(xmask))  #torch.logical_and(xmask, ymask)
            new_xyz.append(xyz[selectedmask]) 
            
            selectnumpoints = torch.sum(selectedmask).item()
            new_trbf_center.append(torch.rand((selectnumpoints, 1)).cuda())

            assert self.trbfslinit < 1 
            new_trbf_scale.append(self.trbfslinit * torch.ones((selectnumpoints, 1), device="cuda"))
            new_motion.append(torch.zeros((selectnumpoints, 9), device="cuda")) 
            new_omega.append(torch.zeros((selectnumpoints, 4), device="cuda"))
            new_featuret.append(torch.zeros((selectnumpoints, 3), device="cuda"))

        new_xyz = torch.cat(new_xyz, dim=0)
        new_rotation = torch.zeros((new_xyz.shape[0],4), device="cuda")
        new_rotation[:, 1]= 0
        
        new_opacity = inverse_sigmoid(0.1 *torch.ones_like(new_xyz[:, 0:1]))
        new_trbf_center = torch.cat(new_trbf_center, dim=0)
        new_trbf_scale = torch.cat(new_trbf_scale, dim=0)
        new_motion = torch.cat(new_motion, dim=0)
        new_omega = torch.cat(new_omega, dim=0)
        new_featuret = torch.cat(new_featuret, dim=0)
        
        new_mask = torch.ones_like(new_xyz[:, 0:1])
         

        tmpxyz = torch.cat((new_xyz, self._xyz), dim=0)
        dist2 = torch.clamp_min(distCUDA2(tmpxyz), 0.0000001)
        dist2 = dist2[:new_xyz.shape[0]]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)
        new_scaling = scales 


        self.densification_postfix(new_xyz, new_opacity, new_scaling, new_rotation, new_trbf_center, new_trbf_scale, new_motion, new_omega,new_featuret, new_mask)
        return new_xyz.shape[0]




    def prune_pointswithemsmask(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        self._features_t = optimizable_tensors["f_t"]
        self._mask = optimizable_tensors["mask"]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self.maskforems = self.maskforems[valid_points_mask] # we only remain valid mask from ems 
        
    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x
        
    def _sort_morton(self):
        with torch.no_grad():
            xyz_q = (
                (2**21 - 1)
                * (self._xyz - self._xyz.min(0).values)
                / (self._xyz.max(0).values - self._xyz.min(0).values)
            ).long()
            order = mortonEncode(xyz_q).sort().indices
            
            self._xyz = nn.Parameter(self._xyz[order], requires_grad=True)
            self._opacity = nn.Parameter(self._opacity[order], requires_grad=True)
            self._scaling = nn.Parameter(self._scaling[order], requires_grad=True)
            self._rotation = nn.Parameter(self._rotation[order], requires_grad=True)

            self._trbf_center = nn.Parameter(self._trbf_center[order], requires_grad=True)
            self._trbf_scale = nn.Parameter(self._trbf_scale[order], requires_grad=True)
            self._motion = nn.Parameter(self._motion[order], requires_grad=True)
            self._omega = nn.Parameter(self._omega[order], requires_grad=True)
            self._features_t = nn.Parameter(self._features_t[order], requires_grad=True)