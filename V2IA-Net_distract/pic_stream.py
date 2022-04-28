import os
import sys
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from util.util import tensor2im
from data.base_dataset import get_transform 

import PIL
from PIL import Image, ImageOps
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
import cv2
import numpy as np
import time

import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms
from FacePose_pytorch.dectect import AntiSpoofPredict
from FacePose_pytorch.pfld.pfld import PFLDInference, AuxiliaryNet  
from FacePose_pytorch.compute import find_pose, get_num

from resnet_3d_old.opts import parse_opts
from resnet_3d_old.mean import get_mean, get_std
from resnet_3d_old.model_c import generate_model
from resnet_3d_old.spatial_transforms_winbus import (
    Compose, Normalize, RandomHorizontalFlip, ToTensor, RandomVerticalFlip, 
    ColorAugment)
from resnet_3d_old.temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop

import check_status as cs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取動作種類
def read_classes(file_path):

    fp = open(file_path, "r")
    classes = fp.readline()
    classes = classes. split(",")
    fp.close()

    return classes

if __name__ == '__main__':
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 讀取動作種類

    classes = read_classes('classes.txt')
    print(classes)

    cut_opt = TestOptions().parse()  # get test options
    cut_opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    cut_opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    cut_model = create_model(cut_opt)      # create a model given opt.model and other options
    cut_model.setup(cut_opt)               # regular setup: load and print networks; create schedulers
    cut_transform = get_transform(cut_opt)
    ts_b = torch.rand(5, 3)
    if cut_opt.eval:
        cut_model.eval()

    

    final_path = ["C:/Users/hongze/Desktop/distract_dcl_m/test_dataset/test_1/drink/5.jpg",
                    "C:/Users/hongze/Desktop/distract_dcl_m/test_dataset/test_1/normal/3.jpg",
                    "C:/Users/hongze/Desktop/distract_dcl_m/test_dataset/test_1/text_left/50.jpg",
                    "C:/Users/hongze/Desktop/distract_dcl_m/test_dataset/test_1/text_right/45.jpg",
                    "C:/Users/hongze/Desktop/distract_dcl_m/test_dataset/test_1/talk_left/3.jpg",
                    "C:/Users/hongze/Desktop/distract_dcl_m/test_dataset/test_1/talk_right/3.jpg"]
    i = 0;

    for file in final_path:
        img = cv2.imread(file)


        #draw_mat = frame.copy()

        gray_frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        data = {'A': cut_transform(Image.fromarray(gray_frame)).unsqueeze(0),'B':ts_b, 'A_paths': ['doesnt_really_matter'], 'B_paths': ['doesnt_really_matter']} 

        cut_model.set_input(data)  # unpack data from data loader
        cut_model.test()           # run inference
        visuals = cut_model.get_current_visuals()  # get image results

        im_data = list(visuals.items())[1][1] # grabbing the important part of the result
        cg_im = tensor2im(im_data)  # convert tensor to image

        img = cv2.resize(img, (256,256))
        print(img.shape)
        print(cg_im.shape)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cg_im = cv2.cvtColor(cg_im,cv2.COLOR_BGR2GRAY)
        cv2.putText(img,"Real",(10,30), font, 1,(0),2)
        cv2.putText(cg_im,"Fake",(10,30), font, 1,(0),2)

        print(img.shape)
        print(cg_im.shape)

        image_c = cv2.hconcat([img, cg_im])
        cv2.imwrite("./"+str(i)+".jpg",image_c)
        #cv2.imwrite("./"+"ir_"+str(i)+".jpg",cg_im)
        i = i + 1
