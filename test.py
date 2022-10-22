# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# --------------------------|
import os
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import math

from utils.metric_util import per_class_iu, fast_hist_crop, fast_ups_crop
from dataloader.pc_dataset import get_label_name, get_label_inv_name, update_config
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
import torch.nn.functional as F

from utils.load_save_util import load_checkpoint
from utils.ups import enable_dropout

import warnings

from torch.nn.parallel import DistributedDataParallel

warnings.filterwarnings("ignore")


def save_predictions_sematicKitti(predict_labels_serialized, predict_prob_serialized, path_to_seq_folder,
                                  path_to_seq_folder_prob, sample_name, challenge=False):
    # dump predictions and probability
    predict_labels_serialized.tofile(path_to_seq_folder + '/' + sample_name + '.label')

    if not challenge:
        if not os.path.exists(path_to_seq_folder_prob):
            os.makedirs(path_to_seq_folder_prob)
        predict_prob_serialized.tofile(path_to_seq_folder_prob + '/' + sample_name + '.label')


def save_predictions_wod(predict_labels_serialized, predict_prob_serialized, path_to_seq_folder,
                         path_to_seq_folder_prob, sample_name, challenge=False):
    # dump predictions and probability
    np.save(os.path.join(path_to_seq_folder, sample_name), predict_labels_serialized)

    if not challenge:
        if not os.path.exists(path_to_seq_folder_prob):
            os.makedirs(path_to_seq_folder_prob)
        np.save(os.path.join(path_to_seq_folder_prob, sample_name), predict_prob_serialized)


def main(args):
    os.environ['OMP_NUM_THREADS'] = "1"

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    print(f"distributed: {distributed}")

    pytorch_device = args.local_rank

    if distributed:
        torch.cuda.set_device(pytorch_device)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    config_path = args.config_path

    configs = load_config_data(config_path)

    if args.mode == 'infer' or args.mode == 'val' or args.mode == 'test':
        configs['train_params']['ssl'] = False

    # send config parameters to pc_dataset
    update_config(configs)

    dataset_config = configs['dataset_params']
    dataset_type = 'SemanticKITTI' if 'SemKITTI_sk_multiscan' == dataset_config['pc_dataset_type'] else 'WOD'
    train_dataloader_config = configs['train_data_loader']
    ssl_dataloader_config = configs['ssl_data_loader']
    val_dataloader_config = configs['val_data_loader']
    test_dataloader_config = configs['test_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']
    ssl_batch_size = ssl_dataloader_config['batch_size']
    test_batch_size = test_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    past_frame = train_hypers['past']
    future_frame = train_hypers['future']

    T_past_frame = train_hypers['T_past']
    T_future_frame = train_hypers['T_future']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    SemKITTI_label_name = get_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]
    print(unique_label_str)

    SemKITTI_learningmap_inv = get_label_inv_name(dataset_config["label_mapping"])
    model = model_builder.build(model_config).to(pytorch_device)
    print(f"model_load_path: {model_load_path}")
    if os.path.exists(model_load_path):
        model = load_checkpoint(model_load_path, model, map_location=pytorch_device)
        print(f" loading model_load_path: {model_load_path}")

    # if args.mgpus:
    #     my_model = nn.DataParallel(my_model)
    #     #my_model.cuda()
    # #my_model.cuda()

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[pytorch_device],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    optimizer = optim.Adam(model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader, test_dataset_loader, ssl_dataset_loader = data_builder.build(
        dataset_config,
        train_dataloader_config,
        val_dataloader_config,
        test_dataloader_config,
        ssl_dataloader_config,
        grid_size=grid_size,
        train_hypers=train_hypers)

    # test and validation
    if args.mode == 'val':
        dataset_loader = val_dataset_loader
        batch_size = val_batch_size
        path_to_save_predicted_labels = val_dataloader_config['data_path']  # "val_result"
    elif args.mode == 'test':
        dataset_loader = test_dataset_loader
        batch_size = test_batch_size
        path_to_save_predicted_labels = test_dataloader_config['data_path']  # "test_result"
    elif args.mode == 'infer':
        dataset_loader = ssl_dataset_loader
        batch_size = ssl_batch_size
        path_to_save_predicted_labels = ssl_dataloader_config['data_path']  # "pseudo_label_result"

    # mode to eval
    model.eval()

    # if uncertainty is used, enable dropout
    if args.ups:
        # enable dropout (mc)
        enable_dropout(model)
        # sample forward pass
        f_pass = 10

    with torch.no_grad():
        ups_hist = []
        hist_list = []
        hist_list_op = []
        ups_count = []

        def validation_inference(vox_label, grid, pt_labs, pt_fea, ref_st_idx=None, ref_end_idx=None, lcw=None):
            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in grid]

            if args.ups:
                ups_out_prob = []
                for _ in range(f_pass):
                    predict_labels_raw = model(val_pt_fea_ten, val_grid_ten, batch_size)
                    ups_out_prob.append(F.softmax(predict_labels_raw, dim=1))  # for selecting positive pseudo-labels

                ups_out_prob = torch.stack(ups_out_prob)
                out_std = torch.std(ups_out_prob, dim=0)
                predict_probablity = torch.mean(ups_out_prob, dim=0)
                predict_labels = torch.argmax(predict_probablity, dim=1)

                # keep dimension during finding maximum
                predict_prob_max, predict_prob_ind = torch.max(predict_probablity, dim=1, keepdim=True)

                # squeeze (remove the 1 size form the tensor)
                predict_prob_max = torch.squeeze(predict_prob_max)

                # get the uncertainty of the most probable prediction
                max_std = out_std.gather(1, predict_prob_ind)

                # squeeze (remove the 1 size form the tensor)
                max_std = torch.squeeze(max_std)

            else:
                predict_labels_raw = model(val_pt_fea_ten, val_grid_ten, batch_size)
                predict_labels = torch.argmax(predict_labels_raw, dim=1)
                predict_probablity = torch.nn.functional.softmax(predict_labels_raw, dim=1)
                predict_prob_max, predict_prob_ind = predict_probablity.max(dim=1)

            # move to cpu and detach to convert to numpy
            predict_labels = predict_labels.cpu().detach().numpy()
            predict_probabilitys = predict_prob_max.cpu().detach().numpy()
            if args.ups:
                model_uncertintys = max_std.cpu().detach().numpy()

            for count, i_val_grid in enumerate(grid):

                if args.save_raw:
                    predict_raw = predict_labels_raw[count, grid[count][:, 0], grid[count][:, 1], grid[count][:, 2]]

                predict_label = predict_labels[count, grid[count][:, 0], grid[count][:, 1], grid[count][:, 2]]

                predict_prob = predict_probabilitys[count, grid[count][:, 0], grid[count][:, 1], grid[count][:, 2]]

                if args.ups:
                    model_uncertainty = model_uncertintys[
                        count, grid[count][:, 0], grid[count][:, 1], grid[count][:, 2]]
                    model_uncertainty_serialized = np.array(model_uncertainty, dtype=np.float32)

                predict_labels_serialized = np.array(predict_label, dtype=np.int32)
                predict_prob_serialized = np.array(predict_prob, dtype=np.float32)
                if args.save_raw:
                    predict_raw_serialized = np.array(predict_raw, dtype=np.float32)

                demo_pt_labs = pt_labs[count]

                # get reference frame start and end index
                st_id, end_id = int(ref_st_idx[count]), int(ref_end_idx[count])

                # only select the reference frame points
                if ref_st_idx is not None:
                    predict_labels_serialized = predict_labels_serialized[st_id:st_id + end_id]
                    predict_prob_serialized = predict_prob_serialized[st_id:st_id + end_id]
                    if args.save_raw:
                        predict_raw_serialized = predict_raw_serialized[st_id:st_id + end_id]
                    demo_pt_labs = demo_pt_labs[st_id:st_id + end_id]
                    if args.ups:
                        model_uncertainty_serialized = model_uncertainty_serialized[st_id:st_id + end_id]

                if args.mode == 'val':
                    hist_list.append(fast_hist_crop(predict_labels_serialized, demo_pt_labs,
                                                    unique_label))
                    if args.ups:
                        tmp_hist, temp_count = fast_ups_crop(model_uncertainty_serialized, demo_pt_labs.flatten(),
                                                             unique_label)
                        ups_hist.append(tmp_hist)
                        ups_count.append(temp_count)

                if args.save:

                    # convert the prediction into corresponding GT labels (inverse mapping)
                    # for index, label in enumerate(predict_labels_serialized):
                    #     predict_labels_serialized[index] = SemKITTI_learningmap_inv[label]
                    # print(predict_labels_serialized.size)
                    predict_labels_serialized = np.vectorize(SemKITTI_learningmap_inv.__getitem__)(predict_labels_serialized)

                    # get frame and sequence name
                    sample_name = dataset_loader.dataset.point_cloud_dataset.im_idx[i_iter_val * batch_size + count][
                                  -10:-4]
                    sequence_num = dataset_loader.dataset.point_cloud_dataset.im_idx[i_iter_val * batch_size + count].split('/')[-3]

                    # create destination path to save predictions
                    # path_to_seq_folder = path_to_save_predicted_labels + '/' + str(sequence_num)
                    path_to_seq_folder = os.path.join(path_to_save_predicted_labels, str(sequence_num),
                                                      f"predictions_f{T_past_frame}_{T_future_frame}")
                    path_to_seq_folder_prob = os.path.join(path_to_save_predicted_labels, str(sequence_num),
                                                           f"probability_f{T_past_frame}_{T_future_frame}")
                    if args.save_raw:
                        path_to_seq_folder_raw = os.path.join(path_to_save_predicted_labels, str(sequence_num),
                                                              f"raw_f{T_past_frame}_{T_future_frame}")

                    if args.challenge:
                        path_to_save_test_predicted_labels = args.challenge_path
                        path_to_seq_folder = os.path.join(path_to_save_test_predicted_labels,
                                                          f"f{T_past_frame}_{T_future_frame}", "sequences",
                                                          str(sequence_num),
                                                          "predictions")

                    if not os.path.exists(path_to_seq_folder):
                        os.makedirs(path_to_seq_folder)

                    # dump predictions and probability
                    predict_labels_serialized.tofile(path_to_seq_folder + '/' + sample_name + '.label')

                    if dataset_type == 'SemanticKITTI':
                        save_predictions_sematicKitti(predict_labels_serialized, predict_prob_serialized,
                                                      path_to_seq_folder, path_to_seq_folder_prob, sample_name, challenge=args.challenge)

                    elif dataset_type == 'WOD':
                        save_predictions_wod(predict_labels_serialized, predict_prob_serialized,
                                             path_to_seq_folder, path_to_seq_folder_prob, sample_name, challenge=args.challenge)

                    else:
                        raise Exception(f'{dataset_type} dataset type not known')

                    # if not args.challenge:
                    #     if not os.path.exists(path_to_seq_folder_prob):
                    #         os.makedirs(path_to_seq_folder_prob)
                    #     predict_prob_serialized.tofile(path_to_seq_folder_prob + '/' + sample_name + '.label')

                    # if args.save_raw:
                    #     if not os.path.exists(path_to_seq_folder_raw):
                    #         os.makedirs(path_to_seq_folder_raw)
                    #
                    #     predict_prob_serialized.tofile(path_to_seq_folder_raw + '/' + sample_name + '.label')

        # Validation with multi-frames and ssl:
        # if past_frame > 0 and train_hypers['ssl']:
        for i_iter_val, (_, vox_label, grid, pt_labs, pt_fea, ref_st_idx, ref_end_idx, lcw) in tqdm(
                enumerate(dataset_loader),
                total=math.ceil(len(dataset_loader.dataset.point_cloud_dataset.im_idx) / batch_size)):
            # call the validation and inference with
            validation_inference(vox_label, grid, pt_labs, pt_fea, ref_st_idx=ref_st_idx, ref_end_idx=ref_end_idx,
                                 lcw=lcw)

        # print the validation per class iou and overall miou
        if args.mode == 'val':
            iou = per_class_iu(sum(hist_list))
            print('Validation per class iou: ')
            for class_name, class_iou in zip(unique_label_str, iou):
                print('%s : %.2f%%' % (class_name, class_iou * 100))
            val_miou = np.nanmean(iou) * 100

            print('Current val miou is %.3f' % val_miou)
        if args.ups:
            uncertainty_hist = np.sum(ups_hist, axis=0) / np.sum(ups_count, axis=0)
            plt.bar(range(20), uncertainty_hist, width=0.4)
            plt.show()
            print(uncertainty_hist)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path',
                        default='config/semantickitti/semantickitti_S0_0_T11_33_ssl_s20_p80.yaml')
    parser.add_argument('-g', '--mgpus', action='store_true', default=False)
    parser.add_argument('-m', '--mode', default='val')
    parser.add_argument('-s', '--save', default=True)
    parser.add_argument('-c', '--challenge', default=False)
    parser.add_argument('-p', '--challenge_path', default='/mnt/personal/gebreawe/Datasets/RealWorld/semantic-kitti'
                                                  '/challenge')
    parser.add_argument('-u', '--ups', default=False)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('-r', '--save_raw', default=False)
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
