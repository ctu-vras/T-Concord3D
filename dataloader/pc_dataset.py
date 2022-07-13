# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# --------------------------|

import os
import numpy as np
from torch.utils import data
import yaml
import pickle
from os.path import exists
import glob
from os.path import join

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
                 return_ref=True, label_mapping="semantic-kitti.yaml", demo_label_path=None):
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


@register_dataset
class SemKITTI_sk_multiscan(data.Dataset):
    def __init__(self, data_path, imageset='train',return_ref=False, label_mapping="semantic-kitti-multiscan.yaml", nusc=None, wod=None, ssl_data_path=None):
        global past, future, ssl, T_past, T_future
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.data_path = data_path
        if imageset == 'train':
            self.split = semkittiyaml['split']['train']
            if ssl and (ssl_data_path is not None):
                self.split += semkittiyaml['split']['pseudo']
        elif imageset == 'val':
            self.split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            self.split = semkittiyaml['split']['test']
        elif imageset == 'pseudo':
            self.split = semkittiyaml['split']['pseudo']
        else:
            raise Exception('Split must be train/val/test')

        self.multiscan = past  # 2 -additional two frames are fused with target-frame. Hence, 3 point clouds in total
        self.past = past
        self.future = future
        self.T_past = T_past
        self.T_future = T_future
        self.im_idx = []

        self.calibrations = []
        self.times = []
        self.poses = []

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
        self.times = []
        self.poses = {}  # []

        for seq in self.split:
            seq_folder = join(self.data_path, str(seq).zfill(2))

            # Read Calib
            self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

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

    def fuse_multi_scan(self, points, pose0, pose):

        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        # new_points = hpoints.dot(pose.T)
        new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

        new_points = new_points[:, :3]
        new_coords = new_points - pose0[:3, 3]
        # new_coords = new_coords.dot(pose0[:3, :3])
        new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
        new_coords = np.hstack((new_coords, points[:, 3:]))

        return new_coords

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        origin_len = len(raw_data)
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            if ssl and exists(self.im_idx[index].replace('velodyne', f"predictions_f{self.T_past}_{self.T_future}")[:-3] + 'label'):
                annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', f"predictions_f{self.T_past}_{self.T_future}")[:-3] + 'label', dtype=np.int32).reshape((-1, 1))
            else:
                annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.int32).reshape((-1, 1))

            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            if ssl and exists(self.im_idx[index].replace('velodyne', f"probability_f{self.T_past}_{self.T_future}")[:-3] + 'label'):
                lcw = np.fromfile(self.im_idx[index].replace('velodyne', f"probability_f{self.T_past}_{self.T_future}")[:-3] + 'label',
                dtype=np.float32).reshape((-1,1))
                # TODO: check casting
                lcw = (lcw * 100).astype(np.int32)
            elif ssl:  # in case of GT label give weight = 1.0 per label
                lcw = np.expand_dims(np.ones_like(raw_data[:, 0], dtype=np.float32), axis=1)
                # TODO: check casting
                lcw = (lcw * 100).astype(np.int32)

        number_idx = int(self.im_idx[index][-10:-4])
        # dir_idx = int(self.im_idx[index][-22:-20])
        dir_idx = self.im_idx[index].split('/')[-3]

        pose0 = self.poses[dir_idx][number_idx]
        past_frame_len = 0
        future_frame_len = 0
        # TODO: added the future frame availability check
        if self.multiscan and (number_idx - self.multiscan >= 0) \
                and (number_idx + self.multiscan < len(self.poses[dir_idx])):

            for fuse_idx in range(self.multiscan):

                # TODO: past frames
                past_idx = fuse_idx + 1

                pose = self.poses[dir_idx][number_idx - past_idx]

                newpath2 = self.im_idx[index][:-10] + str(number_idx - past_idx).zfill(6) + self.im_idx[index][-4:]
                raw_data2 = np.fromfile(newpath2, dtype=np.float32).reshape((-1, 4))

                # count number of past frame points
                past_frame_len += len(raw_data2)

                if self.imageset == 'test':
                    annotated_data2 = np.expand_dims(np.zeros_like(raw_data2[:, 0], dtype=int), axis=1)
                else:
                    if ssl and exists(newpath2.replace('velodyne', f"predictions_f{self.T_past}_{self.T_future}")[:-3] + 'label'):
                        annotated_data2 = np.fromfile(newpath2.replace('velodyne', f"predictions_f{self.T_past}_{self.T_future}")[
                            :-3] + 'label', dtype=np.int32).reshape((-1, 1))
                    else:
                        annotated_data2 = np.fromfile(newpath2.replace('velodyne', 'labels')[:-3] + 'label',
                                                  dtype=np.int32).reshape((-1, 1))
                    annotated_data2 = annotated_data2 & 0xFFFF  # delete high 16 digits binary
                    if ssl and exists(newpath2.replace('velodyne', f"probability_f{self.T_past}_{self.T_future}")[:-3] + 'label'):
                        lcw2 = np.fromfile(newpath2.replace('velodyne', f"probability_f{self.T_past}_{self.T_future}")[:-3] + 'label',
                                          dtype=np.float32).reshape((-1, 1))
                        # TODO: check casting
                        lcw2 = (lcw2 * 100).astype(np.int32)
                    elif ssl: # in case of GT label give weight = 1.0 per label
                        lcw2 = np.expand_dims(np.ones_like(raw_data2[:, 0], dtype=np.float32), axis=1)
                        # TODO: check casting
                        lcw2 = (lcw2 * 100).astype(np.int32)

                raw_data2 = self.fuse_multi_scan(raw_data2, pose0, pose)

                # past frames
                if len(raw_data2) != 0:
                    raw_data = np.concatenate((raw_data, raw_data2), 0)
                    annotated_data = np.concatenate((annotated_data, annotated_data2), 0)
                    if ssl:
                        lcw = np.concatenate((lcw, lcw2), 0)

                # TODO: future frames
                if self.future > 0:
                    future_idx = fuse_idx + 1

                    pose = self.poses[dir_idx][number_idx + future_idx]

                    newpath3 = self.im_idx[index][:-10] + str(number_idx + future_idx).zfill(6) + self.im_idx[index][-4:]
                    raw_data3 = np.fromfile(newpath3, dtype=np.float32).reshape((-1, 4))

                    # count number of future frame points
                    future_frame_len += len(raw_data3)

                    if self.imageset == 'test':
                        annotated_data3 = np.expand_dims(np.zeros_like(raw_data3[:, 0], dtype=int), axis=1)
                    else:
                        if ssl and exists(newpath3.replace('velodyne', f"predictions_f{self.T_past}_{self.T_future}")[:-3] + 'label'):
                            annotated_data3 = np.fromfile(
                                newpath3.replace('velodyne', f"predictions_f{self.T_past}_{self.T_future}")[
                                :-3] + 'label', dtype=np.int32).reshape((-1, 1))
                        else:
                            annotated_data3 = np.fromfile(newpath3.replace('velodyne', 'labels')[:-3] + 'label',
                                                      dtype=np.int32).reshape((-1, 1))
                        annotated_data3 = annotated_data3 & 0xFFFF  # delete high 16 digits binary
                        if ssl and exists(newpath3.replace('velodyne', f"probability_f{self.T_past}_{self.T_future}")[:-3] + 'label'):
                            lcw3 = np.fromfile(newpath3.replace('velodyne', f"probability_f{self.T_past}_{self.T_future}")[:-3] + 'label',
                                              dtype=np.float32).reshape((-1, 1))
                            # TODO: check casting
                            lcw3 = (lcw3 * 100).astype(np.int32)


                        elif ssl: # in case of GT label give weight = 1.0 per label
                            lcw3 = np.expand_dims(np.ones_like(raw_data3[:, 0], dtype=np.float32), axis=1)
                            # TODO: check casting
                            lcw3 = (lcw3 * 100).astype(np.int32)

                    raw_data3 = self.fuse_multi_scan(raw_data3, pose0, pose)

                    # TODO: check correctness (future frame)
                    if len(raw_data3) != 0:
                        raw_data = np.concatenate((raw_data3, raw_data, ), 0)
                        annotated_data = np.concatenate((annotated_data3, annotated_data), 0)
                        if ssl:
                            lcw = np.concatenate((lcw3, lcw),0)

        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))

        if self.return_ref and ssl:
            # TODO: masking below 0.8 confidence
            # lcw_mask = lcw < 80
            # lcw[lcw_mask] = 0
            data_tuple += (raw_data[:, 3], lcw, future_frame_len, origin_len) # origin_len is used to indicate the length of target-scan and lcw
        elif self.return_ref:
            data_tuple += (raw_data[:, 3], future_frame_len, origin_len) # origin_len is used to indicate the length of target-scan

        return data_tuple

# WOD ----------------------


# load Semantic KITTI class info
def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name


def get_SemKITTI_label_inv_name(label_inv_mapping):
    with open(label_inv_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_inv_name = semkittiyaml['learning_map_inv']

    return SemKITTI_label_inv_name


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
