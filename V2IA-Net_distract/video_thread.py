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

import check_status as cs
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取動作種類
def read_classes(file_path):

    fp = open(file_path, "r")
    classes = fp.readline()
    classes = classes. split(",")
    fp.close()

    return classes

# 臉部資訊轉換 : 臉部座標資訊作為輸入，轉換成特徵點偵測所需要的輸入格式
def crop_range(x1, x2, y1, y2, w, h):

    size = int(max([w, h]))
    cx = x1 + w/2
    cy = y1 + h/2
    x1 = int(cx - size/2)
    x2 = int(x1 + size)
    y1 = int(cy - size/2)
    y2 = int(y1 + size)

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)

    return x1, x2, y1, y2, dx, dy, edx, edy

# GUI選檔介面，選取測試影片，回傳影片路徑
class Qt(QWidget):

    def mv_Chooser(self):

        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "D:/new_daytime/distract/crop","Mp4 (*.mp4)", options=opt)
    
        return fileUrl[0]

# V2IA-Net二元分類種類
dn = ['distract', 'normal']
# V2IA-Net動作辨識種類
classes = read_classes('classes.txt')
print(classes)

# 輸出資訊
point_dict = {}          # 臉部特徵點座標
distract_output = ""     # 動作辨識種類
predict_dn_output = ""   # 二元分類種類
fake_ir_image = []
yaw = ""                 # yaw角度
pitch = ""               # pitch角度
roll = ""                # roll角度

face = False             # 判斷是否有偵測到臉部

# 駕駛行為分析 (V2IA-Net)
def driverAction(mat, model):
    # 調整輸入影像格式
    gray_frame = cv2.cvtColor(mat,cv2.COLOR_BGR2GRAY)
    data = {'A': transform(Image.fromarray(gray_frame)).unsqueeze(0), 'A_paths': ['doesnt_really_matter']} 
    # 設定輸入
    model.set_input(data)  
    # 模型計算
    model.test()
    # 拿取動作辨識種類輸出
    actionType = model.get_predict()['predict_A']
    action_output = classes[actionType.argmax()]
    # 拿取二元分類種類輸出
    binaryType = model.get_predict()['predict_A_dn']
    binary_output = dn[binaryType.argmax()]

    visuals = model.get_current_visuals()
    im_data = list(visuals.items())[1][1] # grabbing the important part of the result
    cg_im = tensor2im(im_data)  # convert tensor to image
    
    print(cg_im)
    global distract_output
    global predict_dn_output
    global fake_ir_image

    distract_output = action_output
    predict_dn_output = binary_output
    fake_ir_image = cg_im
# 頭部姿態分析
def headPosture(mat, faceModel, landmarkModel, trans):
    # 臉部偵測
    image_bbox = faceModel.get_bbox(mat)
    face_x1 = image_bbox[0]
    face_y1 = image_bbox[1]
    face_x2 = image_bbox[0] + image_bbox[2]
    face_y2 = image_bbox[1] + image_bbox[3]
    face_w = face_x2 - face_x1
    face_h = face_y2 - face_y1

    crop_x1, crop_x2, crop_y1, crop_y2, dx, dy, edx, edy = crop_range(face_x1, face_x2, face_y1, face_y2, face_w, face_h)
        
    # 從原始影像裁切臉部區域   
    cropped = mat[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]
    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
        cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
    ratio_w = face_w / 112
    ratio_h = face_h / 112

    # 縮放影像112x112
    cropped = cv2.resize(cropped, (112, 112))
    face_input = cropped.copy()
    face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
    face_input = trans(face_input).unsqueeze(0).to(device)

    # 臉部特徵點預測
    _, landmarks = landmarkModel(face_input)
    pre_landmark = landmarks[0]
    pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]

    # 紀錄特徵點座標
    pointDict = {}
    i = 0
    
    for (x,y) in pre_landmark.astype(np.float32):
        pointDict[f'{i}'] = [x,y]
        # 繪製特徵點
        # cv2.circle(draw_mat,(int(face_x1 + x * ratio_w),int(face_y1 + y * ratio_h)), 2, (255, 0, 0), -1)
        i += 1

    # 判斷是否有偵測到臉部
    global face 
    if(face_w < 20 or face_h < 20):
        face = False
    else:
        face = True
    
    global yaw, pitch, roll

    # 分析當前影像頭部各軸狀態
    yaw, pitch, roll = find_pose(pointDict)

    global point_dict 
    point_dict = pointDict
    

if __name__ == '__main__':
    
    # 利用QT選取輸入影像   
    qt_env = QApplication(sys.argv)
    process = Qt()
    fileUrl = process.mv_Chooser()

    if(fileUrl == ""):
        print("Without input file!!")
        sys.exit(0)

    print(fileUrl)

    # 各軸頭部姿態
    left_right = ""
    up_down = "" 
    tilt = ""

    # 駕駛行為狀態
    distract_output = ""
    # 統計計算張數 (10張影像進行一次駕駛頭部姿態分析)
    full_clip = 0      

    # 綜合危險值
    distract_score = 0

    font = cv2.FONT_HERSHEY_SIMPLEX     # opencv顯示字體
    # -------------------------------------------------------------------------------
    # V2IA-Net
    v2ianet_opt = TestOptions().parse()  # get test options
    v2ianet_opt.no_flip = True           # no flip; comment this line if results on flipped images are needed.
    v2ianet_opt.display_id = -1          # no visdom display; the test code saves the results to a HTML file.
    v2ianet_model = create_model(v2ianet_opt)    # create a model given opt.model and other options
    v2ianet_model.setup(v2ianet_opt)             # regular setup: load and print networks; create schedulers
    transform = get_transform(v2ianet_opt)
    
    if v2ianet_opt.eval:
        v2ianet_model.eval()
   
    # -------------------------------------------------------------------------------
    # 頭部姿態分析

    # 載入臉部偵測模型 
    face_model = AntiSpoofPredict(0) # (使用distract/FacePose_pytorch/ 內的程式)

    # 載入特徵點偵測模型 # (使用distract/FacePose_pytorch/ 內的程式)
    headpose_model = './FacePose_pytorch/checkpoint/snapshot/checkpoint.pth.tar'
    checkpoint_h = torch.load(headpose_model, map_location=device)
    plfd_backbone = PFLDInference().to(device)
    plfd_backbone.load_state_dict(checkpoint_h['plfd_backbone'])
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.to(device)
    headpose_transformer = transforms.Compose([transforms.ToTensor()])

    # -------------------------------------------------------------------------------
    # 開始偵測
    cap = cv2.VideoCapture(fileUrl)
    ret, frame = cap.read()

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 儲存結果影片
    videoWriter = cv2.VideoWriter("./result.avi",cv2.VideoWriter_fourcc('X','V','I','D'),fps,(width,height))

    while(ret):

        start = time.time()     
        ret, frame = cap.read()
        if(not ret):
            break
        draw_mat = frame.copy()

        # 駕駛行為分析
        threadA = threading.Thread(target = driverAction, args = (frame, v2ianet_model) )
        threadA.start()

        # 頭部姿態分析
        threadB = threading.Thread(target = headPosture, args = (frame, face_model, plfd_backbone, headpose_transformer) )
        threadB.start()
        
        threadA.join()
        threadB.join()


        # 統計狀態
        cs.headpose.headpose_series(yaw, pitch, roll)

        full_clip = full_clip+1

        # 累計滿10張影像, 累計滿10筆頭部姿態資料
        if (full_clip > 9):

            # ------------------------------------------------------------
            # 頭部姿態變化分析
            left_right, up_down, tilt = cs.headpose.headpose_output()
            print(left_right, up_down, tilt)
            if(not face):
                left_right, up_down, tilt = "", "", ""
                
            # ------------------------------------------------------------
            # 計算綜合危險值
            distract_score = cs.dis_head(distract_output, left_right, up_down, tilt)

            #------------------------------------------------------------
            # 清除暫存資料
            cs.headpose.clear()
            full_clip = 0
        end = time.time()

        
        # 輸出資訊
        print("#####################################################################")
        print("\n")
        print("--------------處理時間--------------")
        print('Time       : ', round(end-start,3), '(s)')
        print("\n")
        print("--------------預測資訊--------------")
        print('駕駛行為分析 : ', distract_output)
        print('Yaw角度      : ', yaw)
        print('Pitch角度    : ', pitch)
        print('Roll角度     : ', roll)
        print('左右轉動     : ', left_right)
        print('上下仰俯     : ', up_down)
        print('左右歪斜     : ', tilt)
        print('危險值       : ', distract_score)
        print("\n")
        print("----------------警示----------------")
        if(distract_score >= 35):
            print('Yes')
        else:
            print('No')

        print("\n\n")
        
        
        # -------------------------------------------------------------------------------
        # 繪製畫面 (紅色表示異常)
        color_normal = ((255,0,0)) 
        color_abnormal = ((0,0,255))    
        cv2.rectangle(draw_mat, (0, 0), (230, 230), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.rectangle(draw_mat, (700, 0), (920, 40), (255, 255, 255), -1, cv2.LINE_AA)

        # cv2.rectangle(draw_mat, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 255), 2, cv2.LINE_AA)

        # -------------------------------------------------------------------------------
        # 頭部姿態分析
        cv2.putText(draw_mat,"R-L",(10,65), font,0.8,(255,0,0),2) 
        cv2.putText(draw_mat,"U-D",(10,95), font,0.8,(255,0,0),2)
        cv2.putText(draw_mat,"TILT",(10,125), font,0.8,(255,0,0),2)

        cv2.putText(draw_mat, ": ", (95,65), font, 0.8, (255,0,0), 2)
        cv2.putText(draw_mat, ": ", (95,95), font, 0.8, (255,0,0), 2)
        cv2.putText(draw_mat, ": ", (95,125), font, 0.8, (255,0,0), 2)

        if (str(left_right) == 'normal'):
            cv2.putText(draw_mat, str(left_right), (110,65), font, 0.8, color_normal, 2)
        else:
            cv2.putText(draw_mat, str(left_right), (110,65), font, 0.8, color_abnormal, 2)
        if (str(up_down) == 'normal'):
            cv2.putText(draw_mat, str(up_down), (110,95), font, 0.8, color_normal, 2)
        else:
            cv2.putText(draw_mat, str(up_down), (110,95), font, 0.8, color_abnormal, 2)
        if (str(tilt) == 'normal'):
            cv2.putText(draw_mat, str(tilt), (110,125), font, 0.8, color_normal, 2)
        else:
            cv2.putText(draw_mat, str(tilt), (110,125), font, 0.8, color_abnormal, 2)
        
        # -------------------------------------------------------------------------------
        # 駕駛行為分析
        cv2.putText(draw_mat,"Status",(10,35), font,0.8,(255,0,0),2) 
        cv2.putText(draw_mat,": ",(95,35), font, 0.6,(255,0,0),2)

        if(distract_output == 'normal'):
            cv2.putText(draw_mat, distract_output,(110,35), font, 0.8, color_normal, 2)
        else:
            cv2.putText(draw_mat, distract_output,(110,35), font, 0.8, color_abnormal, 2)
        # -------------------------------------------------------------------------------
        # 駕駛二元分類
        cv2.putText(draw_mat,"Binary :",(710,30), font,0.8,(255,0,0),2) 
        if(predict_dn_output == 'normal'):
            cv2.putText(draw_mat, predict_dn_output,(815,30), font, 0.8, color_normal, 2)
        else:
            cv2.putText(draw_mat, predict_dn_output,(815,30), font, 0.8, color_abnormal, 2)
        
        # -------------------------------------------------------------------------------
        # 分心警示
        if(distract_score >= 35):
            cv2.rectangle(draw_mat, (10, 170), (220, 225), (120, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(draw_mat,"dangerous!",(20,210), font, 1,(120,0,255),2,cv2.LINE_AA)
        
        cv2.putText(draw_mat,"FPS     : "+str(int(1/((end-start)+0.000001))),(10,155), font, 0.6,(0,0,0),2)
        
        fake_ir_image = cv2.resize(fake_ir_image, (width,height))
        cv2.putText(fake_ir_image,"V2IA Fake IR Image",(30,30), font, 1,(0,0,255),2)
        image_c = cv2.hconcat([draw_mat, fake_ir_image])

        cv2.imshow("draw_mat", image_c)
        
        videoWriter.write(draw_mat)

        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            videoWriter.release()
            break
        
    videoWriter.release()
    cap.release()
