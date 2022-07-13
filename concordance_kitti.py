# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# --------------------------|

import os
import time
import argparse
import sys

import numpy as np
import glob
import os
import shutil
import random
import math


# https://github.com/ctu-vras/T-Concord3D.git


def main(args):
    teacher_1 = args.teacher1
    teacher_2 = args.teacher2
    teacher_3 = args.teacher3
    lamda = args.lamda
    concordance = args.concordance
    source = args.source
    destination = args.destination
    sequence = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]

    for i, sq in enumerate(sequence):
        pred_t1 = sorted(glob.glob(os.path.join(source, sq, f"predictions_{teacher_1}", '*.label')))
        pred_t2 = sorted(glob.glob(os.path.join(source, sq, f"predictions_{teacher_2}", '*.label')))

        probs_t1 = sorted(glob.glob(os.path.join(source, sq, f"probability_{teacher_1}", '*.label')))
        probs_t2 = sorted(glob.glob(os.path.join(source, sq, f"probability_{teacher_2}", '*.label')))
        if teacher_3 is not None:
            pred_t3 = sorted(glob.glob(os.path.join(source, sq, f"predictions_{teacher_3}", '*.label')))
            probs_t3 = sorted(glob.glob(os.path.join(source, sq, f"probability_{teacher_3}", '*.label')))

        frame_len = len(pred_t1)

        for frame in range(frame_len):
            frame_name = str(frame).zfill(6)
            if teacher_3 is not None:
                pred = np.array([np.fromfile(pred_t1[frame], dtype=np.int32).reshape((-1, 1)),
                                 np.fromfile(pred_t2[frame], dtype=np.int32).reshape((-1, 1)),
                                 np.fromfile(pred_t3[frame], dtype=np.int32).reshape((-1, 1))])
                prob = np.array([np.fromfile(probs_t1[frame], dtype=np.float32).reshape((-1, 1)),
                                 np.fromfile(probs_t2[frame], dtype=np.float32).reshape((-1, 1)),
                                 np.fromfile(probs_t3[frame], dtype=np.float32).reshape((-1, 1))])
            else:
                pred = np.array([np.fromfile(pred_t1[frame], dtype=np.int32).reshape((-1, 1)),
                                 np.fromfile(pred_t2[frame], dtype=np.int32).reshape((-1, 1))])
                prob = np.array([np.fromfile(probs_t1[frame], dtype=np.float32).reshape((-1, 1)),
                                 np.fromfile(probs_t2[frame], dtype=np.float32).reshape((-1, 1))])

            max_pob = prob.max(axis=0)
            max_pob_id = prob.argmax(axis=0)
            best_pred = np.zeros_like(pred[0])
            for j in range(len(max_pob)):
                best_pred[j] = pred[int(max_pob_id[j]), j]

            weight = np.zeros_like(best_pred)

            for k in range(3):
                predicted = pred[k]
                concord = best_pred == predicted
                weight += concord.astype(int)

            best_prob = max_pob
            new_prob = best_prob + ((weight - 1) * lamda)
            new_prob = np.minimum(np.ones_like(best_prob), new_prob)
            new_prob = new_prob.astype(np.float32)

            if not os.path.exists(os.path.join(destination, sq, f"predictions_{concordance}")):
                os.makedirs(os.path.join(destination, sq, f"predictions_{concordance}"))
                os.makedirs(os.path.join(destination, sq, f"probability_{concordance}"))

            best_pred.tofile(os.path.join(destination, sq, f"predictions_{concordance}", frame_name + '.label'))
            new_prob.tofile(os.path.join(destination, sq, f"probability_{concordance}", frame_name + '.label'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lamda', default=0.1, type=float)
    parser.add_argument('-b', '--best', default=True, )
    parser.add_argument('-x', '--teacher1', required=False, default="f1_1")
    parser.add_argument('-y', '--teacher2', required=False, default="f2_2")
    parser.add_argument('-z', '--teacher3', default="f3_3")
    parser.add_argument('-c', '--concordance', default="11_33")
    parser.add_argument('-s', '--source', default='/mnt/beegfs/gpu/argoverse-tracking-all-training/semantic-kitti'
                                                  '/train_pseudo_20/sequences')
    parser.add_argument('-d', '--destination', default='/mnt/beegfs/gpu/argoverse-tracking-all-training/semantic'
                                                       '-kitti/train_pseudo_20/sequences')
    args = parser.parse_args()

    main(args)
