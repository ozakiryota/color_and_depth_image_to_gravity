from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import torch
from torchvision import transforms

class DataTransform():
    def __init__(self, resize, mean, std):
        self.resize = resize
        self.mean = mean
        self.std = std
        self.img_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, color_img_pil, depth_img_numpy, acc_numpy, phase="train"):
        ## augemntation
        if phase == "train":
            ## mirror
            is_mirror = bool(random.getrandbits(1))
            if is_mirror:
                ## color
                color_img_pil = ImageOps.mirror(color_img_pil)
                ## depth
                depth_img_numpy = np.fliplr(depth_img_numpy)
                ## acc
                acc_numpy[1] = -acc_numpy[1]
        ## color: pil -> tensor
        color_img_tensor = self.img_transform(color_img_pil)
        ## depth: numpy -> tensor
        depth_img_numpy = depth_img_numpy.astype(np.float32)
        # depth_img_numpy = np.where(depth_img_numpy > 0.0, 1.0/depth_img_numpy, depth_img_numpy)
        # depth_img_numpy = np.clip(depth_img_numpy, 0, 1)
        depth_img_tensor = torch.from_numpy(depth_img_numpy)
        depth_img_tensor = depth_img_tensor.unsqueeze_(0)
        # depth_img_tensor = transforms.functional.normalize(depth_img_tensor, ([0.5]), ([0.5]))
        ## acc: numpy -> tensor
        acc_numpy = acc_numpy.astype(np.float32)
        acc_numpy = acc_numpy / np.linalg.norm(acc_numpy)
        acc_tensor = torch.from_numpy(acc_numpy)
        return color_img_tensor, depth_img_tensor, acc_tensor

##### test #####
# ## color image
# color_img_path = "../../../dataset_image_to_gravity/AirSim/example/camera_0.jpg"
# color_img_pil = Image.open(color_img_path)
# ## depth image
# depth_img_path = "../../../dataset_image_to_gravity/AirSim/example/lidar.npy"
# depth_img_numpy = np.load(depth_img_path)
# print("depth_img_numpy = ", depth_img_numpy)
# print("depth_img_numpy.shape = ", depth_img_numpy.shape)
# ## label
# acc_list = [0, 1, 0]
# acc_numpy = np.array(acc_list)
# print("acc_numpy = ", acc_numpy)
# ## trans param
# resize = 112
# mean = ([0.5, 0.5, 0.5])
# std = ([0.5, 0.5, 0.5])
# ## transform
# transform = DataTransform(resize, mean, std)
# color_img_trans, depth_img_trans, acc_trans = transform(color_img_pil, depth_img_numpy, acc_numpy, phase="train")
# print("acc_trans = ", acc_trans)
# ## tensor -> numpy
# depth_img_trans_numpy = depth_img_trans.numpy().transpose((1, 2, 0))  #(ch, h, w) -> (h, w, ch)
# print("depth_img_trans_numpy.shape = ", depth_img_trans_numpy.shape)
# color_img_trans_numpy = color_img_trans.numpy().transpose((1, 2, 0))  #(ch, h, w) -> (h, w, ch)
# color_img_trans_numpy = np.clip(color_img_trans_numpy, 0, 1)
# print("color_img_trans_numpy.shape = ", color_img_trans_numpy.shape)
# ## save
# depth_img_trans_numpy = depth_img_trans_numpy.squeeze(2)
# depth_img_trans_pil = Image.fromarray(np.uint8(255*depth_img_trans_numpy/np.max(depth_img_trans_numpy)))
# save_depth_path = "../../save/transform_depth.jpg"
# depth_img_trans_pil.save(save_depth_path)
# print("saved: ", save_depth_path)
# color_img_trans_pil = Image.fromarray(np.uint8(255*color_img_trans_numpy))
# save_color_path = "../../save/transform_color.jpg"
# color_img_trans_pil.save(save_color_path)
# print("saved: ", save_color_path)
# ## imshow
# h = 4
# w = 1
# plt.figure()
# plt.subplot(h, w, 1)
# plt.imshow(depth_img_numpy)
# plt.subplot(h, w, 2)
# plt.imshow(color_img_pil)
# plt.subplot(h, w, 3)
# plt.imshow(depth_img_trans_numpy)
# plt.subplot(h, w, 4)
# plt.imshow(color_img_trans_numpy)
# plt.show()
