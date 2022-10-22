# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# --------------------------|

import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_label_name, update_config
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings

from torch.nn.parallel import DistributedDataParallel

warnings.filterwarnings("ignore")

# training
epoch = 0
best_val_miou = 0
global_iter = 0


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

    # send config parameters to pc_dataset
    update_config(configs)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']
    ssl_dataloader_config = configs['ssl_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    past_frame = train_hypers['past']
    future_frame = train_hypers['future']
    ssl = train_hypers['ssl']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    SemKITTI_label_name = get_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config).to(pytorch_device)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model, map_location=pytorch_device)

    # if args.mgpus:
    #     my_model = nn.DataParallel(my_model)
    #     #my_model.cuda()
    # #my_model.cuda()

    if distributed:
        my_model = DistributedDataParallel(
            my_model,
            device_ids=[pytorch_device],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    # for weighted class loss
    weighted_class = False

    # for focal loss
    focal_loss = False  # True

    # 20 class number of samples from training sample

    class_weights = np.array([1.40014903e+00, 1.10968683e+00, 5.06321920e+02, 9.19710291e+01,
                              1.76627589e+01, 1.58902791e+01, 1.49002594e+02, 6.12058299e+02,
                              1.75137027e+03, 2.47504075e-01, 3.25237847e+00, 3.62211985e-01,
                              8.77872638e+00, 4.08248861e-01, 9.97997655e-01, 1.91585640e-01,
                              7.21493239e+00, 4.68076958e-01, 1.69628483e+01, 6.35032127e+01], dtype=np.float32)

    per_class_weight = None
    if focal_loss or weighted_class:
        per_class_weight = torch.from_numpy(class_weights).to(pytorch_device)

    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    if ssl:
        loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                       num_class=num_class, ignore_label=ignore_label,
                                                       weights=per_class_weight, ssl=True, fl=focal_loss)
    else:
        loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                       num_class=num_class, ignore_label=ignore_label,
                                                       weights=per_class_weight, fl=focal_loss)

    train_dataset_loader, val_dataset_loader, _, _ = data_builder.build(dataset_config,
                                                                     train_dataloader_config,
                                                                     val_dataloader_config,
                                                                     ssl_dataloader_config=ssl_dataloader_config,
                                                                     grid_size=grid_size,
                                                                     train_hypers=train_hypers)

    class_count = np.zeros(20)
    my_model.train()
    # global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    global global_iter, best_val_miou, epoch
    print("|-------------------------Training started-----------------------------------------|")
    print(f"focal_loss:{focal_loss}, weighted_cross_entropy: {weighted_class}")

    while epoch < train_hypers['max_num_epochs']:
        print(f"epoch: {epoch}")
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(5)

        # lr_scheduler.step(epoch)

        def valideting(hist_list, val_loss_list, val_vox_label, val_grid, val_pt_labs, val_pt_fea, ref_st_idx=None,
                       ref_end_idx=None, lcw=None):
            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

            predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
            # aux_loss = loss_fun(aux_outputs, point_label_tensor)

            inp = val_label_tensor.size(0)

            # TODO: check if this is correctly implemented
            # hack for batch_size mismatch with the number of training example
            predict_labels = predict_labels[:inp, :, :, :, :]

            if ssl:
                lcw_tensor = torch.FloatTensor(lcw).to(pytorch_device)
                loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                      ignore=ignore_label, lcw=lcw_tensor) + loss_func(predict_labels.detach(),
                                                                                       val_label_tensor, lcw=lcw_tensor)
            else:
                loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                      ignore=ignore_label) + loss_func(predict_labels.detach(), val_label_tensor)

            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()
            for count, i_val_grid in enumerate(val_grid):
                hist_list.append(fast_hist_crop(predict_labels[
                                                    count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                    val_grid[count][:, 2]], val_pt_labs[count],
                                                unique_label))
            val_loss_list.append(loss.detach().cpu().numpy())

            return hist_list, val_loss_list

        # if global_iter % check_iter == 0 and epoch >= 1:
        if epoch >= 1:
            my_model.eval()
            hist_list = []
            val_loss_list = []
            with torch.no_grad():
                # Validation with multi-frames and ssl:
                # if past_frame > 0 and train_hypers['ssl']:
                for i_iter_val, (_, vox_label, grid, pt_labs, pt_fea, ref_st_idx, ref_end_idx, val_lcw) \
                        in enumerate(val_dataset_loader):
                    # call the validation and inference with
                    hist_list, val_loss_list = valideting(hist_list, val_loss_list, vox_label, grid, pt_labs,
                                                          pt_fea, ref_st_idx=ref_st_idx,
                                                          ref_end_idx=ref_end_idx,
                                                          lcw=val_lcw)

                print(f"--------------- epoch: {epoch} ----------------")

                iou = per_class_iu(sum(hist_list))
                print('Validation per class iou: ')
                for class_name, class_iou in zip(unique_label_str, iou):
                    print('%s : %.2f%%' % (class_name, class_iou * 100))
                val_miou = np.nanmean(iou) * 100
                # del val_vox_label, val_grid, val_pt_fea

                # save model if performance is improved
                if best_val_miou < val_miou:
                    best_val_miou = val_miou
                    if not os.path.exists(model_save_path.split('/')[-2]):
                        os.mkdir(os.path.join(model_save_path.split('/')[-2]))
                    torch.save(my_model.state_dict(), model_save_path)

                print(f"Current val miou is {np.round(val_miou, 2)} while the best val miou is "
                      f"{np.round(best_val_miou, 2)}")
                print(f"Current val loss is {np.round(np.mean(val_loss_list), 2)}")

        def training(i_iter_train, train_vox_label, train_grid, pt_labels, train_pt_fea, ref_st_idx=None,
                     ref_end_idx=None, lcw=None):
            global global_iter, best_val_miou, epoch

            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)

            # forward + backward + optimize
            outputs = my_model(train_pt_fea_ten, train_vox_ten, train_batch_size)
            inp = point_label_tensor.size(0)
            # print(f"outputs.size() : {outputs.size()}")
            # TODO: check if this is correctly implemented
            # hack for batch_size mismatch with the number of training example
            outputs = outputs[:inp, :, :, :, :]
            ################################

            if ssl:
                lcw_tensor = torch.FloatTensor(lcw).to(pytorch_device)
                loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=ignore_label,
                                      lcw=lcw_tensor) + loss_func(
                    outputs, point_label_tensor, lcw=lcw_tensor)
            else:
                loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor,
                                      ignore=ignore_label) + loss_func(
                    outputs, point_label_tensor)

            # TODO: check --> to mitigate only one element tensors can be converted to Python scalars
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter_train, np.mean(loss_list)))
                else:
                    print('loss error')

            optimizer.zero_grad()

            global_iter += 1

            if global_iter % 100 == 0:
                pbar.update(100)

            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter_train, np.mean(loss_list)))
                else:
                    print('loss error')

        my_model.train()
        # training with multi-frames and ssl:
        # if past_frame > 0 and train_hypers['ssl']:
        for i_iter_train, (_, vox_label, grid, pt_labs, pt_fea, ref_st_idx, ref_end_idx, lcw) in enumerate(
                train_dataset_loader):
            # call the validation and inference with
            training(i_iter_train, vox_label, grid, pt_labs, pt_fea, ref_st_idx=ref_st_idx, ref_end_idx=ref_end_idx,
                     lcw=lcw)

        pbar.close()
        epoch += 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path',
                        default='config/semantickitti/nuscenes_S0_0_T11_33_ssl_s20_p80.yaml')
    parser.add_argument('-g', '--mgpus', action='store_true', default=False)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
