import csv
import os
import numpy as np
import math

def makeDataList(rootpath, csv_name, num_cameras=-1):
    csv_path = os.path.join(rootpath, csv_name)
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        data_list = []
        for row in reader:
            if num_cameras < 0:
                num_cameras = len(row[4:])
            ## depth
            row[3] = os.path.join(rootpath, row[3])
            for i in range(4, 4 + num_cameras):
                row[i] = os.path.join(rootpath, row[i])
                camera_angle = 2*math.pi/len(row[4:])*(i - 4)
                rot_acc_list = rotateVector(row[:3], camera_angle)
                data = rot_acc_list + [row[3], row[i], camera_angle]
                data_list.append(data)
    return data_list

def rotateVector(acc_list, camera_angle):
    acc_numpy = np.array(acc_list)
    acc_numpy = acc_numpy.astype(np.float32)
    rot = np.array([
        [math.cos(-camera_angle), -math.sin(-camera_angle), 0.0],
        [math.sin(-camera_angle), math.cos(-camera_angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    rot_acc_numpy = np.dot(rot, acc_numpy)
    rot_acc_list = rot_acc_numpy.tolist()
    return rot_acc_list

##### test #####
# rootpath = "../../../dataset_image_to_gravity/AirSim/lidar4cam/train"
# csv_name = "imu_lidar_camera.csv"
# train_list = makeDataList(rootpath, csv_name)
# # print(train_list)
# print("example0: ", train_list[0][:3], train_list[0][3:])
# print("example1: ", train_list[1][:3], train_list[1][3:])
# print("example1: ", train_list[2][:3], train_list[2][3:])
