from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import sys
sys.path.append('../')
from common import trainer_mod
from common import make_datalist_mod
from common import data_transform_mod
from common import dataset_mod
from common import network_mod
import criterion_mod

class Trainer(trainer_mod.Trainer):
    def saveGraph(self, record_loss_train, record_loss_val):    #overwrite
        graph = plt.figure()
        plt.plot(range(len(record_loss_train)), record_loss_train, label="Training")
        plt.plot(range(len(record_loss_val)), record_loss_val, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss [m/s^2]")
        plt.title("loss: train=" + str(record_loss_train[-1]) + ", val=" + str(record_loss_val[-1]))
        graph.savefig("../../graph/" + self.str_hyperparameter + ".jpg")
        plt.show()

def main():
    ## hyperparameters
    method_name = "mle"
    train_rootpath = "../../../dataset_image_to_gravity/AirSim/lidar1cam/train"
    val_rootpath = "../../../dataset_image_to_gravity/AirSim/lidar1cam/val"
    csv_name = "imu_lidar_camera.csv"
    resize = 112
    mean_element = 0.5
    std_element = 0.5
    optimizer_name = "Adam"  #"SGD" or "Adam"
    lr_colorcnn = 1e-5
    lr_depthcnn = 1e-5
    lr_fc = 1e-4
    batch_size = 100
    num_epochs = 50
    ## dataset
    train_dataset = dataset_mod.OriginalDataset(
        data_list=make_datalist_mod.makeDataList(train_rootpath, csv_name),
        transform=data_transform_mod.DataTransform(
            resize,
            ([mean_element, mean_element, mean_element]),
            ([std_element, std_element, std_element])
        ),
        phase="train"
    )
    val_dataset = dataset_mod.OriginalDataset(
        data_list=make_datalist_mod.makeDataList(val_rootpath, csv_name),
        transform=data_transform_mod.DataTransform(
            resize,
            ([mean_element, mean_element, mean_element]),
            ([std_element, std_element, std_element])
        ),
        phase="val"
    )
    ## network
    net = network_mod.Network(resize, dim_fc_out=9, use_pretrained_vgg=True)
    ## criterion
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = criterion_mod.Criterion(device)
    ## train
    trainer = Trainer(
        method_name,
        train_dataset, val_dataset,
        net, criterion,
        optimizer_name, lr_colorcnn, lr_depthcnn, lr_fc,
        batch_size, num_epochs
    )
    trainer.train()

if __name__ == '__main__':
    main()
