import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import time

import torch
from torchvision import models
import torch.nn as nn

import sys
sys.path.append('../')
from common import inference_mod
from common import make_datalist_mod
from common import data_transform_mod
from common import dataset_mod
from common import network_mod

def main():
    ## hyperparameters
    rootpath = "../../../dataset_image_to_gravity/AirSim/lidar1cam/val"
    csv_name = "imu_lidar_camera.csv"
    resize = 112
    mean_element = 0.5
    std_element = 0.5
    batch_size = 10
    weights_path = "../../weights/regression.pth"
    ## dataset
    dataset = dataset_mod.OriginalDataset(
        data_list=make_datalist_mod.makeDataList(rootpath, csv_name),
        transform=data_transform_mod.DataTransform(
            resize,
            ([mean_element, mean_element, mean_element]),
            ([std_element, std_element, std_element])
        ),
        phase="val"
    )
    ## network
    net = network_mod.Network(resize, dim_fc_out=3, use_pretrained_vgg=False)
    ## criterion
    criterion = nn.MSELoss()
    ## infer
    inference = inference_mod.Inference(
        dataset,
        net, weights_path, criterion,
        batch_size
    )
    inference.infer()

if __name__ == '__main__':
    main()
