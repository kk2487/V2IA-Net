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
from tqdm import tqdm 
import PIL
from PIL import Image, ImageOps
import cv2
import numpy as np
import time

import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取動作種類
def read_classes(file_path):

    fp = open(file_path, "r")
    classes = fp.readline()
    classes = classes. split(",")
    fp.close()

    return classes

if __name__ == '__main__':

    # 讀取動作種類
    font = cv2.FONT_HERSHEY_SIMPLEX
    test_file_path = [
    "./test_dataset/test_n1/",
    "./test_dataset/test_n2/",
    "./test_dataset/test_n3/",
    "./test_dataset/test_n4/",
    "./test_dataset/test_n5/",
    "./test_dataset/test_n6/",
    "./test_dataset/test_n7/",
    "./test_dataset/test_n8/",
    "./test_dataset/test_n9/",
    "./test_dataset/night_test_n1/",
    "./test_dataset/night_test_n2/",
    "./test_dataset/night_test_n3/",
    "./test_dataset/night_test_n4/",
    "./test_dataset/night_test_n5/",
    "./test_dataset/night_test_n6/",
    "./test_dataset/night_test_n7/",
    "./test_dataset/night_test_n8/",
    "./test_dataset/night_test_n9/",
    "./test_dataset/rgbir_dataset76_daytime/",
    "./test_dataset/rgbir_dataset76_night/",
    "./dynamic_test_dataset/daytime_dynamic",
    "./dynamic_test_dataset/night_dynamic",
    ]

    classes = read_classes('classes.txt')
    dn = ['distract', 'normal']
    #print(classes)

    cut_opt = TestOptions().parse()  # get test options
    cut_opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    cut_opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dcl_model = create_model(cut_opt)      # create a model given opt.model and other options
    dcl_model.setup(cut_opt)               # regular setup: load and print networks; create schedulers
    dcl_transform = get_transform(cut_opt)

    ts_b = torch.rand(5, 3)
    if cut_opt.eval:
        dcl_model.eval()

    for t in tqdm(test_file_path): 
        act_dir = os.listdir(t)

        total_num = []
        correct_num = []
        all_total=0
        all_correct=0

        normal_num = 0 
        distract_num = 0
        correct_normal_num = 0 
        correct_distract_num = 0

        for ad in act_dir:
            sec_folder_dir = os.path.join(t, ad)
            #print(sec_folder_dir)

            label = str(ad)
            #print("This is label : ", label)

            img_list = os.listdir(sec_folder_dir)

            total_num.append(len(img_list))
            correct = 0
            for img in img_list: 
                
                if(label == dn[1]):
                    normal_num = normal_num + 1
                    dn_label = dn[1]
                
                else:
                    distract_num = distract_num + 1  
                    dn_label = dn[0]
                

                final_path = os.path.join(sec_folder_dir, img)
                #print(final_path)

                img = cv2.imread(final_path)

                gray_frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                data = {'A': dcl_transform(Image.fromarray(gray_frame)).unsqueeze(0),'B':ts_b, 'A_paths': ['doesnt_really_matter'], 'B_paths': ['doesnt_really_matter'], 'A_label':ts_b, 'B_label':ts_b} 

                dcl_model.set_input(data)  # unpack data from data loader
                dcl_model.test()           # run inference
                visuals = dcl_model.get_current_visuals()  # get image results

                # 拿取DCL模型動作預測結果
                predict_gan = dcl_model.get_predict()['predict_A']

                gan_predict_output = classes[predict_gan.argmax()]
                

                # 拿取DCL模型分心預測結果
                predict_dn = dcl_model.get_predict()['predict_A_dn']
                #print(predict_dn)
                predict_dn_output = dn[predict_dn.argmax()]

                #print(dn_label, predict_dn_output)

                # im_data = list(visuals.items())[1][1] # grabbing the important part of the result
                # cg_im = tensor2im(im_data)  # convert tensor to image
                
                # cg_im = cv2.resize(cg_im, (224,224))

                # cv2.putText(img, gan_predict_output,(40,40), font, 1.5,(0,255,0),2)
                
                # #print(gan_predict_output)
                # img = cv2.resize(img, (224,224))

                # image_c = cv2.hconcat([img, cg_im])
                # image_c = cv2.resize(image_c, (1200,600))
                if(str(gan_predict_output) == str(label)):
                    correct = correct + 1
                
                if(str(predict_dn_output) == dn_label and dn_label == dn[1]):
                    correct_normal_num = correct_normal_num +1 

                if(str(predict_dn_output) == dn_label and dn_label == dn[0]):
                    correct_distract_num = correct_distract_num +1





                # cv2.imshow("image_c", image_c)
                # if cv2.waitKey(1) == 27:
                #     cv2.destroyAllWindows()
                #     break
            correct_num.append(correct)
        print("--------------------------------------")
        print(correct_num) 
        print(total_num)
        print("----------------Accuracy--------------") 
        for i in range(6):
            print(classes[i], ":", round(correct_num[i]/total_num[i]*100, 2), "%")
            all_total = all_total + total_num[i] 
            all_correct = all_correct + correct_num[i]
        print("------------Average Accuracy----------")
        print(round(all_correct/all_total*100, 2), "%")


        print("------------Binary Accuracy----------")
        print(correct_distract_num, correct_normal_num)
        print(distract_num, normal_num)
        print(round(correct_distract_num/distract_num*100, 2), "%", round(correct_normal_num/normal_num*100, 2), "%")

        path = 'output.txt'
        with open(path, 'a') as f:
            
            f.write('--------------------------------------\n')
            f.write('Test Path : '+ t)
            f.write('\n')
            f.write('Test Time : '+ str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

            f.write('\n')
            for i in range(6):
                f.write(str(correct_num[i]))
                f.write(" ")
            f.write('\n')
            for i in range(6):
                f.write(str(total_num[i]))
                f.write(" ")
            f.write('\n')
            f.write("----------------Accuracy--------------\n") 
            for i in range(6):
                f.write(classes[i] + ":" + str(round(correct_num[i]/total_num[i]*100, 2)) + "%" + '\n')
                all_total = all_total + total_num[i] 
                all_correct = all_correct + correct_num[i]
            f.write("Average Accuracy"+ ":" +str(round(all_correct/all_total*100, 2))+ "%\n")
            f.write("-------------Binary Accuracy----------\n") 
            f.write(str(correct_distract_num) + " " + str(correct_normal_num) + '\n')
            f.write(str(distract_num) + " " + str(normal_num) + '\n')
            f.write(str(round(correct_distract_num/distract_num*100, 2)) + "%" + " " + str(round(correct_normal_num/normal_num*100, 2)) + "%\n")

            f.write('--------------------------------------\n')
            f.write('\n')