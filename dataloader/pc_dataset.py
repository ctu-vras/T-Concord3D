# -*- coding:utf-8 -*-
# author: Xinge
# @file: pc_dataset.py

import glob
import os
import pickle
from os.path import exists

import numpy as np
import yaml
from torch.utils import data

REGISTERED_PC_DATASET_CLASSES = {}

# past and future frames global place holders
past = 0
future = 0
T_past = 0
T_future = 0
ssl = False
rgb = False


def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]


@register_dataset
class SemKITTI_demo(data.Dataset):
    def __init__(self, data_path, imageset='demo',
                 return_ref=True, label_mapping="semantic-wod.yaml", demo_label_path=None):
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.return_ref = return_ref

        self.im_idx = []
        self.im_idx += absoluteFilePaths(data_path)
        self.label_idx = []
        if self.imageset == 'val':
            print(demo_label_path)
            self.label_idx += absoluteFilePaths(demo_label_path)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'demo':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        elif self.imageset == 'val':
            annotated_data = np.fromfile(self.label_idx[index], dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple


@register_dataset
class SemKITTI_sk(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="semantic-wod.yaml", nusc=None, ssl_data_path=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        global past, future, ssl, T_past, T_fture
        self.past = past
        self.future = future
        self.T_past = T_past
        self.T_future = T_future

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        '''
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple
        '''
        origin_len = len(raw_data)
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            # x = self.im_idx[index].replace('velodyne', f"predictions_{self.T_past}_{self.T_future}")[:-3] + 'label'
            if ssl and exists(self.im_idx[index].replace('velodyne', f"predictions_{self.T_past}_{self.T_future}")[
                              :-3] + 'label'):
                annotated_data = np.fromfile(
                    self.im_idx[index].replace('velodyne', f"predictions_{self.T_past}_{self.T_future}")[
                    :-3] + 'label',
                    dtype=np.int32).reshape((-1, 1))
            else:
                annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                             dtype=np.int32).reshape((-1, 1))

            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            # annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            if ssl and exists(self.im_idx[index].replace('velodyne', f"probability_{self.T_past}_{self.T_future}")[
                              :-3] + 'label'):
                lcw = np.fromfile(self.im_idx[index].replace('velodyne', f"probability_{self.T_past}_{self.T_future}")[
                                  :-3] + 'label',
                                  dtype=np.float32).reshape((-1, 1))
                # TODO: check casting
                lcw = (lcw * 100).astype(np.int32)
            elif ssl:  # in case of GT label give weight = 1.0 per label
                lcw = np.expand_dims(np.ones_like(raw_data[:, 0], dtype=np.float32), axis=1)
                # TODO: check casting
                lcw = (lcw * 100).astype(np.int32)

        number_idx = int(self.im_idx[index][-10:-4])
        dir_idx = int(self.im_idx[index][-22:-20])

        past_frame_len = 0
        future_frame_len = 0

        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))

        if self.return_ref and ssl:
            # np.save('pcl.npy', raw_data[:, :3])
            # np.save('label.npy', annotated_data)

            # TODO: masking below 0.8 confidence
            # lcw_mask = lcw < 80
            # lcw[lcw_mask] = 0

            data_tuple += (raw_data[:, 3], lcw, future_frame_len,
                           origin_len)  # origin_len is used to indicate the length of target-scan and lcw

        elif self.return_ref:
            data_tuple += (
                raw_data[:, 3], future_frame_len,
                origin_len)  # origin_len is used to indicate the length of target-scan

        return data_tuple


@register_dataset
class SemKITTI_nusc(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="nuscenes.yaml", nusc=None):
        self.return_ref = return_ref

        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.nusc_infos = data['infos']
        self.data_path = data_path
        self.nusc = nusc

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'][16:]
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                self.nusc.get('lidarseg', lidar_sd_token)['filename'])

        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        data_tuple = (points[:, :3], points_label.astype(np.uint8))
        if self.return_ref:
            data_tuple += (points[:, 3],)
        return data_tuple


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    remove_ind = label == 0
    label -= 1
    label[remove_ind] = 255
    return label


from os.path import join


def transform_pcl_scan(points, pose0, pose):
    # pose = poses[0][idx]

    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
    # new_points = hpoints.dot(pose.T)
    new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

    new_points = new_points[:, :3]
    new_coords = new_points - pose0[:3, 3]
    # new_coords = new_coords.dot(pose0[:3, :3])
    new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
    new_coords = np.hstack((new_coords, points[:, 3:]))

    return new_coords


def fuse_multiscan(ref_raw_data, ref_annotated_data, ref_lcw, transformed_data,
                   transformed_annotated_data, transformed_lcw, source, ssl):
    lcw = None
    if (source != 1) and (source != -1):
        print(f"Error data source {source} not Implemented")
        return 0
    if source == -1:  # past frame
        raw_data = np.concatenate((transformed_data, ref_raw_data), 0)
        annotated_data = np.concatenate((transformed_annotated_data, ref_annotated_data), 0)
        if ssl:
            lcw = np.concatenate((transformed_lcw, ref_lcw), 0)

    if source == 1:  # future frame
        raw_data = np.concatenate((ref_raw_data, transformed_data,), 0)
        annotated_data = np.concatenate((ref_annotated_data, transformed_annotated_data), 0)
        if ssl:
            lcw = np.concatenate((ref_lcw, transformed_lcw), 0)

    return raw_data, annotated_data, lcw


def get_combined_data(raw_data, annotated_data, lcw, learning_map, return_ref, origin_len, preceding_frame_len, ssl):
    #print(np.unique(annotated_data))
    annotated_data = np.vectorize(learning_map.__getitem__)(annotated_data)

    data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))

    if return_ref and ssl:
        # np.save('pcl.npy', raw_data[:, :3])
        # np.save('label.npy', annotated_data)

        # TODO: masking below 0.8 confidence
        # lcw_mask = lcw < 80
        # lcw[lcw_mask] = 0

        # origin_len is used to indicate the length of target-scan and lcw
        data_tuple += (raw_data[:, 3], lcw, preceding_frame_len, origin_len)

    elif return_ref:
        # origin_len is used to indicate the length of target-scan
        data_tuple += (raw_data[:, 3], preceding_frame_len, origin_len)

    return data_tuple


@register_dataset
class SemKITTI_sk_multiscan(data.Dataset):
    def __init__(self, data_path, imageset='train', return_ref=False, label_mapping="semantic-kitti-multiscan.yaml",
                 train_hypers=None, wod=None, ssl_data_path=None):
        global past, future, ssl, T_past, T_fture
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)

        self.return_ref = return_ref
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.data_path = data_path
        self.past = train_hypers['past']
        self.future = train_hypers['future']
        self.T_past = train_hypers['T_past']
        self.T_future = train_hypers['T_future']
        self.ssl = train_hypers['ssl']
        self.im_idx = []

        self.calibrations = []
        # self.times = []
        self.poses = []

        if imageset == 'train':
            self.split = semkittiyaml['split']['train']
            if self.ssl and (ssl_data_path is not None):
                self.split += semkittiyaml['split']['pseudo']
        elif imageset == 'val':
            self.split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            self.split = semkittiyaml['split']['test']
        elif imageset == 'pseudo':
            self.split = semkittiyaml['split']['pseudo']
        else:
            raise Exception(f'{imageset}: Split must be train/val/test/pseudo')

        if self.past or self.future:
            self.load_calib_poses()

        for i_folder in self.split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        # self.times = []
        self.poses = {}  # []

        for seq in self.split:
            seq_folder = join(self.data_path, str(seq).zfill(2))

            # Read Calib
            self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            # self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.calibrations[-1])
            # self.poses.append([pose.astype(np.float32) for pose in poses_f64])
            self.poses[seq] = [pose.astype(np.float32) for pose in poses_f64]

    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        # print(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)
        # print(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def get_semantickitti_data(self, newpath, time_frame_idx):
        raw_data = np.fromfile(newpath, dtype=np.float32).reshape((-1, 4))
        lcw = None
        if self.imageset == 'test' or self.imageset == 'pseudo':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            if self.ssl and exists(newpath.replace('velodyne', f"predictions_{self.T_past}_{self.T_future}")[:-3]
                                   + 'label'):
                annotated_data = np.fromfile(
                    newpath.replace('velodyne', f"predictions_{self.T_past}_{self.T_future}")[:-3] + 'label',
                    dtype=np.int64).reshape((-1, 1))
            else:
                annotated_data = np.fromfile(newpath.replace('velodyne', 'labels')[:-3] + 'label',
                                             dtype=np.int32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            # if np.sum(np.unique(annotated_data == 18)) > 0:
            #     print(newpath)

            if self.ssl and exists(newpath.replace('velodyne', f"probability_{self.T_past}_{self.T_future}")[
                                   :-3] + 'label'):
                lcw = np.fromfile(
                    newpath.replace('velodyne', f"probability_{self.T_past}_{self.T_future}")[
                    :-3] + 'label',
                    dtype=np.float32).reshape((-1, 1))
                # TODO: check casting
                lcw = (lcw * 100).astype(np.int32)

            elif self.ssl:  # in case of GT label give weight = 1.0 per label
                lcw = np.expand_dims(np.ones_like(raw_data[:, 0], dtype=np.float32), axis=1)
                # TODO: check casting
                lcw = (lcw * 100).astype(np.int32)

        return raw_data, annotated_data, len(raw_data), lcw

    def __getitem__(self, index):
        # reference scan
        reference_file = self.im_idx[index]
        raw_data, annotated_data, data_len, lcw = self.get_semantickitti_data(reference_file, 0)

        origin_len = data_len

        number_idx = int(self.im_idx[index][-10:-4])
        # dir_idx = int(self.im_idx[index][-22:-20])
        dir_idx = self.im_idx[index].split('/')[-3]

        # past scan
        past_frame_len = 0
        # TODO: added the future frame availability check
        if self.past and ((number_idx - self.past) >= 0) and ((number_idx + self.past) < len(self.poses[dir_idx])):
            # extract the poss of the reference frame
            pose0 = self.poses[dir_idx][number_idx]
            for fuse_idx in range(self.past):
                # TODO: past frames
                frame_ind = fuse_idx + 1
                pose = self.poses[dir_idx][number_idx - frame_ind]
                past_file = self.im_idx[index][:-10] + str(number_idx - frame_ind).zfill(6) + self.im_idx[index][-4:]
                past_raw_data, past_annotated_data, past_data_len, past_lcw = self.get_semantickitti_data(past_file,
                                                                                                          -frame_ind)

                past_raw_data = transform_pcl_scan(past_raw_data, pose0, pose)

                # past frames
                if past_data_len != 0:
                    raw_data, annotated_data, lcw = fuse_multiscan(raw_data, annotated_data, lcw,
                                                                   past_raw_data, past_annotated_data, past_lcw, -1,
                                                                   self.ssl)
                    # count number of past frame points
                    past_frame_len += past_data_len

        # future scan
        future_frame_len = 0
        # TODO: added the future frame availability check
        if self.future and ((number_idx - self.future) >= 0) and (
                (number_idx + self.future) < len(self.poses[dir_idx])):
            # extract the poss of the reference frame
            pose0 = self.poses[dir_idx][number_idx]
            for fuse_idx in range(self.future):
                # TODO: future frame
                frame_ind = fuse_idx + 1
                future_pose = self.poses[dir_idx][number_idx + frame_ind]
                future_file = self.im_idx[index][:-10] + str(number_idx + frame_ind).zfill(6) + self.im_idx[index][-4:]
                future_raw_data, future_annotated_data, future_data_len, future_lcw = self.get_semantickitti_data(
                    future_file, frame_ind)

                future_raw_data = transform_pcl_scan(future_raw_data, pose0, future_pose)

                # TODO: check correctness (future frame)
                if future_data_len != 0:
                    raw_data, annotated_data, lcw = fuse_multiscan(raw_data, annotated_data, lcw,
                                                                   future_raw_data, future_annotated_data, future_lcw,
                                                                   1, self.ssl)
                    # count number of future frame points
                    future_frame_len += future_data_len

        # extract compiled data_tuple
        data_tuple = get_combined_data(raw_data, annotated_data, lcw, self.learning_map, self.return_ref,
                                       origin_len, past_frame_len, self.ssl)

        # # TODO: added the future frame availability check

        return data_tuple


# WOD -------------------------------------------------------------
# def fuse_multi_scan(points, pose0, pose):
#     # pose = poses[0][idx]
#
#     hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
#     # new_points = hpoints.dot(pose.T)
#     new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
#
#     new_points = new_points[:, :3]
#     new_coords = new_points - pose0[:3, 3]
#     # new_coords = new_coords.dot(pose0[:3, :3])
#     new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
#     new_coords = np.hstack((new_coords, points[:, 3:]))
#
#     return new_coords


@register_dataset
class WOD_multiscan(data.Dataset):
    def __init__(self, data_path, imageset='train', return_ref=False, label_mapping="wod-multiscan_labelled.yaml",
                 train_hypers=None, wod=None, ssl_data_path=None):
        global past, future, ssl, T_past, T_fture, rgb
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            wodyaml = yaml.safe_load(stream)
        self.learning_map = wodyaml['learning_map']
        self.imageset = imageset
        self.data_path = data_path
        self.past = train_hypers['past']
        self.future = train_hypers['future']
        self.T_past = train_hypers['T_past']
        self.T_future = train_hypers['T_future']
        self.rgb = train_hypers['rgb']
        self.ssl = train_hypers['ssl']
        # self.use_time = train_hypers['time']  # Use time instead of intensity
        # self.UDA = train_hypers['uda']
        self.im_idx = []
        self.calibrations = []
        # self.times = []
        self.poses = {}

        if imageset == 'train':
            self.split = wodyaml['split']['train']
            # self.sensor_zpose = train_hypers["S_sensor_zpose"]
            if self.ssl and (ssl_data_path is not None):
                self.split += wodyaml['split']['pseudo']
        elif imageset == 'val':
            self.split = wodyaml['split']['valid']
            # self.sensor_zpose = train_hypers["S_sensor_zpose"]
        elif imageset == 'test':
            self.split = wodyaml['split']['test']
            # self.sensor_zpose = train_hypers["S_sensor_zpose"]
        elif imageset == 'pseudo':
            self.split = wodyaml['split']['pseudo']
            # self.sensor_zpose = train_hypers["T_sensor_zpose"]
        else:
            raise Exception(f'{imageset}: Split must be train/val/test/pseudo')

        # self.split = sorted(os.listdir(self.data_path))
        # self.training_len = len(self.split)
        # self.ssl_data_path = ssl_data_path   # '/mnt/beegfs/gpu/argoverse-tracking-all-training/WOD/processed/Unlabeled/testing'
        # xx = self.data_path.split("/")[-1]
        # if ssl and self.data_path.split("/")[-1] == "training" and self.ssl_data_path:
        #     self.training_len = len(sorted(os.listdir(self.data_path)))
        #     self.split = sorted(os.listdir(self.data_path)) + sorted(os.listdir(self.ssl_data_path))
        # print(len(self.split))
        # TODO: remove after search experiment
        # self.split = self.split[150:]
        if self.past or self.future:
            self.load_calib_poses()

        # for c, i_folder in enumerate(self.split):
        #     if ssl and (self.data_path.split("/")[-1] == "training") and (
        #             c >= self.training_len):  # 789 number of training folders
        #         self.im_idx += absoluteFilePaths('/'.join([self.ssl_data_path, str(i_folder), 'lidar']))
        #     else:
        #         self.im_idx += absoluteFilePaths('/'.join([self.data_path, str(i_folder), 'lidar']))

        for c, i_folder in enumerate(self.split):
            self.im_idx += absoluteFilePaths('/'.join([self.data_path, str(i_folder), 'lidar']))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def load_calib_poses(self):
        """
        load calib poses and times.
        """
        ###########
        # Load data
        ###########

        self.calibrations = []
        # self.times = []
        self.poses = {}  # []

        for k, seq in enumerate(self.split):  # range(0, 22):
            # if ssl and (self.data_path.split("/")[-1] == "training") and (
            #         k >= self.training_len):  # 789 number of training folders
            #     seq_folder = join(self.ssl_data_path, str(seq))
            # else:
            seq_folder = join(self.data_path, str(seq))

            # Read poses
            poses_f64 = self.parse_poses(seq_folder, k)
            # self.poses.append([pose.astype(np.float32) for pose in poses_f64])
            self.poses[seq] = [pose.astype(np.float32) for pose in poses_f64]

    def parse_poses(self, seq, k):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        filename = sorted(glob.glob(os.path.join(seq, "poses", "*.npy")))

        poses = []

        for file in filename:
            pose = np.load(file)
            poses.append(pose)
        return poses

    def get_wod_data(self, newpath, time_frame_idx):
        raw_data = np.load(newpath)
        # if self.use_time:
        #     raw_data[:, 3] = np.ones_like(raw_data[:, 3]) * time_frame_idx

        # if self.UDA:
        #     raw_data[:, 2] += self.sensor_zpose  # elevate the point cloud two meters up to align with WOD

        # TODO: check if the colors are encoded correctly instead of the lidar intensity
        if self.rgb:
            # load rgb colors for each points
            raw_rgb = np.load(newpath.replace('lidar', 'colors')[
                              :-3] + 'npy')
            # convert rgb into gray scale [0, 255]
            raw_gray = 0.2989 * raw_rgb[:, 0] + 0.5870 * raw_rgb[:, 1] + 0.1140 * raw_rgb[:, 2]
            # mask (0) ignored point colors  (originally not provided on wod rear-cameras) -> rgb:[1,1,1] or gray:[
            # 0.99990])
            gray_mask = raw_gray > 1  # < 1 #0.9998999999999999
            # assign 0 to the place we want to mask
            raw_gray[gray_mask] = -1
            # replace intensity with gray scale camera image/frame color
            raw_data[:, 3] = raw_gray
            # raw_data[:,4] = gray_mask * 1
        lcw = None
        origin_len = len(raw_data)
        if self.imageset == 'test' or self.imageset == 'pseudo':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0]), axis=1).reshape((-1, 1))
        else:
            # x = self.im_idx[index].replace('lidar', f"predictions_{self.T_past}_{self.T_future}")[:-3] + 'label'
            if self.ssl and exists(newpath.replace('lidar', f"predictions_{self.T_past}_{self.T_future}")[
                                   :-3] + 'npy'):
                annotated_data = np.load(
                    newpath.replace('lidar', f"predictions_{self.T_past}_{self.T_future}")[
                    :-3] + 'npy').reshape((-1, 1))
            else:
                # print(self.im_idx[index].replace('lidar', 'labels')[:-3] + 'npy')
                annotated_data = np.load(newpath.replace('lidar', 'labels')[:-3] + 'npy',
                                         allow_pickle=True)
                if len(annotated_data.shape) == 2:
                    if annotated_data.shape[1] == 2:
                        annotated_data = annotated_data[:, 1]
                # Reshape the label/annotation to vector.
                annotated_data = annotated_data.reshape((-1, 1))

            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary

            if self.ssl and exists(newpath.replace('lidar', f"probability_{self.T_past}_{self.T_future}")[
                                   :-3] + 'npy'):
                lcw = np.load(newpath.replace('lidar', f"probability_{self.T_past}_{self.T_future}")[
                              :-3] + 'npy').reshape((-1, 1))
                # TODO: check casting
                lcw = (lcw * 100).astype(np.int32)
            elif self.ssl:  # in case of GT label give weight = 1.0 per label
                lcw = np.expand_dims(np.ones_like(raw_data[:, 0]), axis=1)
                # TODO: check casting
                lcw = (lcw * 100).astype(np.int32)

        return raw_data, annotated_data, len(raw_data), lcw

    def __getitem__(self, index):
        # reference scan
        reference_file = self.im_idx[index]
        raw_data, annotated_data, data_len, lcw = self.get_wod_data(reference_file, 0)

        origin_len = data_len

        number_idx = int(self.im_idx[index][-10:-4])
        # dir_idx = int(self.im_idx[index][-22:-20])
        dir_idx = self.im_idx[index].split('/')[-3]

        # past scan
        past_frame_len = 0
        # TODO: added the future frame availability check
        if self.past and ((number_idx - self.past) >= 0) and ((number_idx + self.past) < len(self.poses[dir_idx])):
            # extract the poss of the reference frame
            pose0 = self.poses[dir_idx][number_idx]
            for fuse_idx in range(self.past):
                # TODO: past frames
                frame_ind = fuse_idx + 1
                pose = self.poses[dir_idx][number_idx - frame_ind]
                past_file = self.im_idx[index][:-10] + str(number_idx - frame_ind).zfill(6) + self.im_idx[index][-4:]
                past_raw_data, past_annotated_data, past_data_len, past_lcw = self.get_wod_data(past_file, -frame_ind)

                # transform the past frame into reference frame coordinate system
                past_raw_data = transform_pcl_scan(past_raw_data, pose0, pose)

                # past frames
                if past_data_len != 0:
                    raw_data, annotated_data, lcw = fuse_multiscan(raw_data, annotated_data, lcw,
                                                                   past_raw_data, past_annotated_data, past_lcw, -1,
                                                                   self.ssl)
                    # count number of past frame points
                    past_frame_len += past_data_len

        # future scan
        future_frame_len = 0
        # TODO: added the future frame availability check
        if self.future and ((number_idx - self.future) >= 0) and (
                (number_idx + self.future) < len(self.poses[dir_idx])):
            # extract the poss of the reference frame
            pose0 = self.poses[dir_idx][number_idx]
            for fuse_idx in range(self.future):
                # TODO: future frame
                frame_ind = fuse_idx + 1
                future_pose = self.poses[dir_idx][number_idx + frame_ind]
                future_file = self.im_idx[index][:-10] + str(number_idx + frame_ind).zfill(6) + self.im_idx[index][-4:]
                future_raw_data, future_annotated_data, future_data_len, future_lcw = self.get_wod_data(future_file,
                                                                                                        frame_ind)

                # transform the future frame into reference frame coordinate system
                future_raw_data = transform_pcl_scan(future_raw_data, pose0, future_pose)

                # TODO: check correctness (future frame)
                if future_data_len != 0:
                    raw_data, annotated_data, lcw = fuse_multiscan(raw_data, annotated_data, lcw,
                                                                   future_raw_data, future_annotated_data, future_lcw,
                                                                   1, self.ssl)
                    # count number of future frame points
                    future_frame_len += future_data_len

        # extract compiled data_tuple
        data_tuple = get_combined_data(raw_data, annotated_data, lcw, self.learning_map, self.return_ref,
                                       origin_len, past_frame_len, self.ssl)

        return data_tuple


# load label class info
def get_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        config_yaml = yaml.safe_load(stream)
    class_label_name = dict()
    for i in sorted(list(config_yaml['learning_map'].keys()))[::-1]:
        class_label_name[config_yaml['learning_map'][i]] = config_yaml['labels'][i]

    return class_label_name


def get_label_inv_name(label_inv_mapping):
    with open(label_inv_mapping, 'r') as stream:
        config_yaml = yaml.safe_load(stream)
    # label_inv_name = dict()
    label_inv_name = config_yaml['learning_map_inv']

    return label_inv_name


def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]

    return nuScenes_label_name


def update_config(configs):
    global past, future, T_past, T_future, ssl, rgb
    train_hypers = configs['train_params']
    past = train_hypers['past']
    future = train_hypers['future']
    T_past = train_hypers['T_past']
    T_future = train_hypers['T_future']
    ssl = train_hypers['ssl']
    rgb = train_hypers['rgb']