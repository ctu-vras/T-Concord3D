# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# --------------------------|
import os
import time
import argparse
import sys
import numpy as np

import numpy as np
import glob
from multiprocessing import Pool
import os
import shutil
import random
import math


def main():
    #sequence = ["04"]
    #des_seq = ["34"]
    sequence = ["00", "01", "02","03", "04", "05", "06", "07", "09", "10"]
    #des_seq = ["11", "12", "13","14", "15", "16", "17", "18", "19", "20"]

    source = '/mnt/beegfs/gpu/argoverse-tracking-all-training/semantic-kitti/train_pseudo_40/sequences'
    destination = '/mnt/beegfs/gpu/argoverse-tracking-all-training/semantic-kitti/train_pseudo_40/sequences'

    for i, sq in enumerate(sequence):
        files = sorted(glob.glob(os.path.join(source, sq, "velodyne", '*.bin')))
        total_frame = range(len(files))


        for frame, data  in enumerate(files):
            frame_name = str(frame).zfill(6)

            frame_data = data[-10:-4]

            os.rename(os.path.join(source, sq, "velodyne", frame_data + '.bin'), os.path.join( source, sq, "velodyne", frame_name + '.bin'))
            os.rename(os.path.join(source, sq, "labels", frame_data + '.label'), os.path.join(source, sq, "labels", frame_name + '.label'))
            # shutil.copy(os.path.join(source, sq, "calib.txt"), os.path.join(destination, des_seq[i], "calib.txt"))
            # shutil.copy(os.path.join(source, sq, "poses.txt"), os.path.join(destination, des_seq[i], "poses.txt"))
            # shutil.copy(os.path.join(source, sq, "times.txt"), os.path.join( destination, des_seq[i], "times.txt"))



if __name__ == '__main__':

    main()
    print(f"------------------------------Task finished-------------------------")
