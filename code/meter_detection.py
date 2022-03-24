# -*- coding: utf-8 -*-
##更新历史：22/3/14 采用数据增强，使得数据集为原来的2倍

from mimetypes import init
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
# Import Mask RCNN
from PIL import Image
import yaml
import imgaug
import imgaug.augmenters as aug

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Root directory of the project
# ROOT_DIR = os.getcwd()

ROOT_DIR = os.path.abspath("../")
print("当前根目录为："+ROOT_DIR)
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn_diou.config import Config
from mrcnn_diou import model as modellib, utils
from mrcnn_diou import visualize
from mrcnn_diou.model import BatchNorm, log

iter_num=0

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 960
    IMAGE_MAX_DIM = 960

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8*7, 16*7, 32*7, 64*7, 128*7)  # anchor side in pixels
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_STRIDE = 1
    RPN_NMS_THRESHOLD = 0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 150

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10

    TRAIN_BN = True
    WEIGHT_DECAY=0.0001




config = ShapesConfig()
config.display()

class MeterDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.safe_load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image,image_id):
        #print("draw_mask-->",image_id)
        #print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        #print("info-->",info)
        #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    #print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    #print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的类别,可以任意添加
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes,可通过这种方式扩展多个物体
        self.add_class("shapes", 1, "meter") # 表盘
        for i in range(count):
            # 获取图片宽和高

            filestr = imglist[i].split(".")[0]
            # print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            # print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]
            mask_path = mask_floder + "/" + filestr + ".png"
            # print("掩码路径为："+mask_path)
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
            # print(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            cv_img = cv2.imdecode(np.fromfile(file=dataset_root_path + "labelme_json/" + filestr + "_json/img.png", dtype=np.uint8), cv2.IMREAD_COLOR)
            # cv_img = cv2.imread(dataset_root_path + "labelme_json\\" + filestr + "_json\\img.png") #Linux格式，即路径全英文时使用

            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # 重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        # print("image_id",image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img,image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("meter") != -1:
                # print "box"
                labels_form.append("meter")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

#基础设置
dataset_root_path=ROOT_DIR+"/dataset/"

val_img_folder=dataset_root_path+"val/img"
train_img_folder = dataset_root_path + "train/img"
train_mask_folder = dataset_root_path + "train/cv2_mask"
val_mask_floder =dataset_root_path+"val/cv2_mask"
#yaml_folder = dataset_root_path
train_imglist = os.listdir(train_img_folder)
val_imglist=os.listdir(val_img_folder)
train_count = len(train_imglist)
val_count=len(val_imglist)



#print("dataset_val-->",dataset_val._image_ids)

# Load and display random samples
#image_ids = np.random.choice(dataset_train.image_ids, 4)
#for image_id in image_ids:
#    image = dataset_train.load_image(image_id)
#    mask, class_ids = dataset_train.load_mask(image_id)
#    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

if __name__=='__main__':
    # 获取所有 GPU 设备列表
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    #train与val数据集准备
    dataset_train = MeterDataset()
    dataset_train.load_shapes(train_count, train_img_folder, train_mask_folder, train_imglist,dataset_root_path+"train/")
    dataset_train.prepare()

    #print("dataset_train-->",dataset_train._image_ids)

    dataset_val = MeterDataset()
    dataset_val.load_shapes(val_count, val_img_folder, val_mask_floder, val_imglist,dataset_root_path+"val/")
    dataset_val.prepare()
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
    # Which weights to start with?
    parser = argparse.ArgumentParser(description='start_point')
    parser.add_argument('--init', type=str, default = "coco")
    args = parser.parse_args()
    init_with = args.init  # imagenet, coco, or last
    print("这次初始化的模型为："+args.init)

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
    # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
    
    my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5),
    ]
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads',
                custom_callbacks=my_callbacks,
                augmentation= aug.Sometimes(5/6,aug.OneOf(
                                            [ 
                                            imgaug.augmenters.Flipud(1), ##上下翻转
                                            imgaug.augmenters.GaussianBlur(),##高斯模糊
                                            imgaug.augmenters.Affine(rotate=(-45, 45)),## 旋转正负45度
                                            imgaug.augmenters.ChannelShuffle(), ##随机通道交换,若采用senet则不能交换
                                            imgaug.augmenters.Affine(scale=(0.5, 1.5))##图像缩小一半，放大1.5倍
                                             ]
                                        )
                                   ))



    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=50,
                layers="all",
                custom_callbacks=my_callbacks,
                augmentation= aug.Sometimes(5/6,aug.OneOf(
                                            [ 
                                            imgaug.augmenters.Flipud(1), ##上下翻转
                                            imgaug.augmenters.GaussianBlur(),##高斯模糊
                                            imgaug.augmenters.Affine(rotate=(-45, 45)),## 旋转正负45度
                                            imgaug.augmenters.ChannelShuffle(), ##随机通道交换
                                            imgaug.augmenters.Affine(scale=(0.5, 1.5))##图像缩小一半，放大1.5倍
                                             ]
                                        )
                                   ))