# -*- coding:utf-8 -*-
"""
SemKITTI dataloader
"""
import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
from torch.utils import data
import pickle

REGISTERED_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_DATASET_CLASSES, f"exist class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_DATASET_CLASSES
    assert name in REGISTERED_DATASET_CLASSES, f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


@register_dataset
class voxel_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, 50, 1.5], min_volume_space=[-50, -50, -3],
                 cut_mix=False):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.flip_aug = flip_aug
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        # TODO check if the cut and mix augmentation is implemented correctly
        self.cut_mix = cut_mix

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        # initialization
        xyz = None
        labels = None
        sig = None
        lcw = None
        ref_st_ind = None
        ref_end_ind = None

        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 5:
            xyz, labels, sig, ref_st_ind, ref_end_ind = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 6:
            xyz, labels, sig, lcw, ref_st_ind, ref_end_ind = data
        else:
            raise Exception('Return invalid data tuple')

        # TODO: check --------------------------------
        # cut mix data augmentation by grabbing instance and add it to a new scene
        if self.cut_mix:
            # load/grab the object
            dir = '/mnt/beegfs/gpu/argoverse-tracking-all-training/WOD/processed/Labeled/cut_mix'
            new_xyz = np.load(f"{dir}/pcl.npy")
            new_label_all = np.load(f"{dir}/ss_id.npy")
            unique_obj = np.unique(new_label_all[:, 0])

            sel_obj_rand = np.random.choice(len(unique_obj), 5)
            new_label = []
            for id in sel_obj_rand:
                obj_mask = new_label_all[:, 0] == id
                new_label.append(new_label_all[obj_mask])

            new_label = np.concatenate(new_label, axis=0)
            # perform random flipping and rotation
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                new_xyz[:, 0] = -new_xyz[:, 0]
            elif flip_type == 2:
                new_xyz[:, 1] = -new_xyz[:, 1]
            elif flip_type == 3:
                new_xyz[:, :2] = -new_xyz[:, :2]

            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            new_xyz[:, :2] = np.dot(new_xyz[:, :2], j)

        xyz = np.concatenate(xyz, new_xyz[:, :3], axis=0)
        labels = np.concatenate(labels, new_label[:, 1], axis=0)

        if sig is not None:
            sig = np.concatenate(sig, new_xyz[:, 3], axis=0)
        if lcw is not None:
            lcw = np.concatenate(lcw, np.ones_like(new_label[:, 1]), axis=0)
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        max_bound = np.percentile(xyz, 100, axis=0)
        min_bound = np.percentile(xyz, 0, axis=0)

        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size

        intervals = crop_range / (cur_grid_size - 1)
        if (intervals == 0).any(): print("Zero interval!")

        grid_ind = (np.floor((np.clip(xyz, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        # TODO: check if there is lcw label confidence weight
        if len(data) == 6:
            # process the lcw
            processed_lcw = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
            lcw_voxel_pair = np.concatenate([grid_ind, lcw], axis=1)
            lcw_voxel_pair = lcw_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
            processed_lcw = nb_process_label(np.copy(processed_lcw), lcw_voxel_pair)

        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) >= 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]),
                                        axis=1)  # np.concatenate((return_xyz, sig), axis=1)#

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)

        if len(data) == 6:
            data_tuple += (processed_lcw, ref_st_ind, ref_end_ind)

        elif len(data) == 5:
            data_tuple += (ref_st_ind, ref_end_ind)

        return data_tuple


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


@register_dataset
class cylinder_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False,
                 transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                 min_rad=-np.pi / 4, max_rad=np.pi / 4,
                 cut_mix=False, use_tta=False):
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.transform = transform_aug
        self.trans_std = trans_std
        self.cut_mix = cut_mix
        self.use_tta = use_tta

        self.noise_rotation = np.random.uniform(min_rad, max_rad)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def rotation_points_single_angle(self, points, angle, axis=0):
        # points: [N, 3]
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        if axis == 1:
            rot_mat_T = np.array(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
                dtype=points.dtype)
        elif axis == 2 or axis == -1:
            rot_mat_T = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=points.dtype)
        elif axis == 0:
            rot_mat_T = np.array(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=points.dtype)
        else:
            raise ValueError("axis should in range")

        return points @ rot_mat_T

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if self.use_tta:
            data_total = []
            voting = 4
            for idx in range(voting):
                data_single_ori = self.get_single_sample(data, index, idx)
                data_total.append(data_single_ori)
            data_total = tuple(data_total)
            return data_total
        else:
            data_single = self.get_single_sample(data, index)
            return data_single

    def get_single_sample(self, data, index, vote_idx=0):
        split = self.point_cloud_dataset.imageset
        # initialization
        xyz = None
        labels = None
        sig = None
        lcw = None
        ref_st_ind = None
        ref_end_ind = None

        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 5:
            xyz, labels, sig, ref_st_ind, ref_end_ind = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 6:
            xyz, labels, sig, lcw, ref_st_ind, ref_end_ind = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)

        else:
            raise Exception('Return invalid data tuple')

        # TODO: check -----------------------------------------------------
        # cut mix data augmentation by grabbing instance and add it to a new scene
        if self.cut_mix and ((split == 'train') or (split == 'ssl')):
            # load/grab the object
            dir = '/mnt/beegfs/gpu/argoverse-tracking-all-training/WOD/processed/Labeled/cut_mix'
            new_xyz_all = np.load(f"{dir}/pcl.npy")
            new_label_all = np.load(f"{dir}/ss_id.npy")
            unique_obj = np.unique(new_label_all[:, 0])
            num_object = 10

            sel_obj_rand = np.random.choice(len(unique_obj), num_object)
            aug_label = []
            aug_xyz = []
            for id in sel_obj_rand:
                obj_mask = new_label_all[:, 0] == id

                new_label = new_label_all[obj_mask]
                new_xyz = new_xyz_all[obj_mask]

                # perform random mix/placement on the road
                road_mask = np.squeeze(labels) == 18
                road_pcl = xyz[road_mask]

                mix_pos_rand = np.random.choice(len(road_pcl), 1)

                mix_position = road_pcl[mix_pos_rand, :]

                mix_p_x = mix_position[:, 0] - 0.5
                mix_p_y = mix_position[:, 1] - 0.5
                mix_p_z = mix_position[:, 2]

                new_xyz[:, 0] = new_xyz[:, 0] - np.max(new_xyz[:, 0])
                new_xyz[:, 1] = new_xyz[:, 1] - np.max(new_xyz[:, 1])
                new_xyz[:, 2] = new_xyz[:, 2] - np.min(new_xyz[:, 2])

                new_xyz[:, 0] = new_xyz[:, 0] + mix_p_x
                new_xyz[:, 1] = new_xyz[:, 1] + mix_p_y
                new_xyz[:, 2] = new_xyz[:, 2] + mix_p_z
                if self.use_tta:
                    flip_type = vote_idx
                else:
                    flip_type = np.random.choice(4, 1)
                if flip_type == 1:
                    new_xyz[:, 0] = -new_xyz[:, 0]
                elif flip_type == 2:
                    new_xyz[:, 1] = -new_xyz[:, 1]

                rotate_rad = np.deg2rad(np.random.random() * 360)
                c, s = np.cos(rotate_rad), np.sin(rotate_rad)
                j = np.matrix([[c, s], [-s, c]])
                new_xyz[:, :2] = np.dot(new_xyz[:, :2], j)

                aug_label.append(new_label)
                aug_xyz.append(new_xyz)

                mframe = int(len(new_xyz) / 130000)
                if mframe > 1:
                    for i in range(1, mframe):
                        new_xyz[:, 0] = new_xyz[:, 0] - i / 2
                        aug_label.append(new_label)
                        aug_xyz.append(new_xyz)

            new_label = np.concatenate(aug_label, axis=0)
            new_xyz = np.concatenate(aug_xyz, axis=0)

            # combine gt data and cut_mix augmentation
            xyz = np.concatenate([xyz, new_xyz[:, :3]], axis=0)
            labels = np.concatenate([labels, new_label[:, 1].reshape(-1, 1)], axis=0)

            if sig is not None:
                sig = np.concatenate([sig.reshape(-1, 1), new_xyz[:, 3].reshape(-1, 1)], axis=0)
                sig = np.squeeze(sig)
            if lcw is not None:
                new_lcw = np.ones_like(new_label[:, 1]) * 100
                lcw = np.concatenate([lcw, new_lcw.reshape(-1, 1)], axis=0)

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            if self.use_tta:
                flip_type = vote_idx
            else:
                flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        # convert coordinate into polar coordinates
        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        # TODO: check if there is lcw label confidence weight
        if len(data) == 6:
            processed_lcw = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
            lcw_voxel_pair = np.concatenate([grid_ind, lcw], axis=1)
            lcw_voxel_pair = lcw_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
            processed_lcw = nb_process_label(np.copy(processed_lcw), lcw_voxel_pair)

        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) >= 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]),
                                        axis=1)  # np.concatenate((return_xyz, sig), axis=1) #

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)

        # include reference frame start and end index
        if len(data) == 6:
            data_tuple += (processed_lcw, ref_st_ind, ref_end_ind)

        # include pseudo label confidence weights
        elif len(data) == 5:
            data_tuple += (ref_st_ind, ref_end_ind)

        return data_tuple


@register_dataset
class polar_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False):
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
        elif len(data) == 5:
            xyz, labels, sig, ref_st_ind, ref_end_ind = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 6:
            xyz, labels, sig, lcw, ref_st_ind, ref_end_ind = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 45) - np.pi / 8
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        # TODO: check if there is lcw label confidence weight
        if len(data) == 6:
            processed_lcw = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
            lcw_voxel_pair = np.concatenate([grid_ind, lcw], axis=1)
            lcw_voxel_pair = lcw_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
            processed_lcw = nb_process_label(np.copy(processed_lcw), lcw_voxel_pair)

        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) >= 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]),
                                        axis=1)  # np.concatenate((return_xyz, sig), axis=1) #

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)

        # include pseudo label confidence weights
        if len(data) == 6:
            data_tuple += (processed_lcw, ref_st_ind, ref_end_ind)

        # refrence frame index
        elif len(data) == 5:
            data_tuple += (ref_st_ind, ref_end_ind)

        return data_tuple


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_lcw(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.float32)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.float32)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def collate_fn_BEV(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    ref_st_index = None
    ref_end_index = None
    lcw2stack = None

    # if multi frame but not ssl: also add the start and end index of reference scan/frame
    if len(data[0]) == 7:
        ref_st_index = [d[5] for d in data]
        ref_end_index = [d[6] for d in data]
        # return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz, ref_st_index, ref_end_index

    # if ssl and multi frame: also add the start and end index of reference scan/frame and
    # confidence probability pseudo label
    elif len(data[0]) == 8:
        ref_st_index = [d[6] for d in data]
        ref_end_index = [d[7] for d in data]
        lcw2stack = np.stack([d[5] for d in data]).astype(np.float32)
        # return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz, ref_st_index, ref_end_index, lcw2stack

    # return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz
    return torch.from_numpy(data2stack), torch.from_numpy(
        label2stack), grid_ind_stack, point_label, xyz, ref_st_index, ref_end_index, lcw2stack


def collate_fn_BEV_tta(data):

    data2stack = np.stack([da2[0] for da1 in data for da2 in da1]).astype(np.float32)
    label2stack = np.stack([da2[1] for da1 in data for da2 in da1]).astype(np.int)

    voxel_label = []
    for da1 in data:
        for da2 in da1:
            voxel_label.append(da2[1])
    #voxel_label.astype(np.int)
    grid_ind_stack = []
    for da1 in data:
        for da2 in da1:
            grid_ind_stack.append(da2[2])
    point_label = []
    for da1 in data:
        for da2 in da1:
            point_label.append(da2[3])
    xyz = []
    for da1 in data:
        for da2 in da1:
            xyz.append(da2[4])
    # index = []
    # for da1 in data:
    #     for da2 in da1:
    #         index.append(da2[5])

    ref_st_index = None
    ref_end_index = None
    lcw2stack = None
    # if multi frame but not ssl: also add the start and end index of reference scan/frame
    if len(data[0]) == 7:
        ref_st_index = [da2[5] for da1 in data for da2 in da1]
        ref_end_index = [da2[6] for da1 in data for da2 in da1]
        # return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz, ref_st_index, ref_end_index

    # if ssl and multi frame: also add the start and end index of reference scan/frame and
    # confidence probability pseudo label
    elif len(data[0]) == 8:
        ref_st_index = [da2[6] for da1 in data for da2 in da1]
        ref_end_index = [da2[7] for da1 in data for da2 in da1]
        lcw2stack = np.stack([da2[5] for da1 in data for da2 in da1]).astype(np.float32)
    # return xyz, voxel_label, grid_ind_stack, point_label, xyz, ref_st_index, ref_end_index, lcw2stack
    return torch.from_numpy(data2stack), torch.from_numpy(
        label2stack), grid_ind_stack, point_label, xyz, ref_st_index, ref_end_index, lcw2stack


def collate_fn_BEV_test(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    index = [d[5] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz, index
