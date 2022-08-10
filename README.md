# Visible to Infrared Action Net (V2IA-Net)
In order to avoid distracted driving, this thesis proposes a real-time driving distraction monitoring system, which uses images to detect whether a driver is distracted. Most of the previous systems can only operate in the daytime environment. We design a system for the daytime and nighttime. It overcomes the difficulty of nighttime detection, and provide a full-time distraction detection service. This system is divided into two main spindles, namely driver behavior analysis and head posture analysis. An action classification model is used to analyze the driver behavior. Angle status of the driver’s head is evaluated and used to analyze head posture. Finally we conduct a comprehensive evaluation to determine whether there is a distraction behavior. We collect a series of visible and infrared distraction images and create a VID Dataset. A translation model, V2IA-Net, is designed. It uses unpaired images to train a GAN model, and visible images can be converted into infrared images.

We also add a classification model architecture into GAN to improve the quality of image transformation through partial supervised learning. At the same time, the characteristics of the model are used to effectively extract the common features of visible and infrared images. It can improve the accuracy of action recognition, and enable driver behavior analysis to have excellent performance in both daytime and nighttime.

![動態日間測試](https://user-images.githubusercontent.com/35215838/183821232-5c376314-5221-4958-b57a-961d5cee1f2a.gif)

## V2IA-Net Structure
![3-5-4_V2IA-Net_detail](https://user-images.githubusercontent.com/35215838/181172618-538c27c4-021e-4af5-80f3-822c520fc49b.png)

In this work, our code is developed based on [DCLGAN](https://github.com/JunlinHan/DCLGAN).
We modify the code { models/dcl_model.py, models/networks.py } to define new generator and loss of : 
* class ResnetGenerator()
* class DCLModel()

## System Structure 
![3-16_系統整合e](https://user-images.githubusercontent.com/35215838/165756454-3a97a001-f40d-4e0b-bf6e-c2e67421781c.png)

## Dependency Package Version
* cuda : 10.1
* pytorch : 1.16.0
* torchvision : 0.7.0
* opencv-python : 4.4.0.46
* dominate : 2.6.0
* scipy : 1.5.4
* visdom==0.1.8.9


## VID Dataset
![image](https://user-images.githubusercontent.com/35215838/172050546-cf589bb3-0fbf-408d-b209-35361f278b0d.png)

### Dataset Description
* This dataset is built with nighttime infrared images and daytime visible images of driver distraction.
* There are six types of driving distractions. Drink、Normal、Talk left、Talk right、Text left、Text right.
* The dataset contains six men and three women.

[VID Dataset]()
/labshare/VID Dataset

```
|--VID Dataset
  |--Video
    |--VID-D_Dataset
    |--VID-M_Dataset
    |--VID-S_Dataset
  |--Training Dataset
    |--rgbir_new_dataset99.zip
  |--Testing Dataset
    |--VID-D_Dataset
    |--VID-M_Dataset
    |--VID-S_Dataset
```

## Usage 
### 1.1 Generate Training Dataset
You can create your own training dataset with this program to label action category.
```
python3 create_train_dataset.py
```
### 1.2 Training Set
We also provide dataset with labeled for training.

[Training Dataset]()
/labshare/VID Dataset/Training Dataset/rgbir_new_dataset99.zip

A : Visible image

B : Infrared image
```
|--dataset
  |--A
    |--drink
    |--normal
    |--talk_left
    |--talk_right
    |--text_left
    |--text_right
  |--B
    |--drink
    |--normal
    |--talk_left
    |--talk_right
    |--text_left
    |--text_right
```

### 1.3 Train
* open visdom server
```
python -m visdom.server
```
* open browser
```
http://localhost:8097/
```
* start training
```
python train.py --dataroot ./datasets/rgbir_new_dataset99 --name rgbir_DCL
```
### 2.1 Generate Testing Dataset
You can create your own testing dataset with this program to label action category.
```
python3 create_test_dataset.py
```
### 2.2 Testing Set
We also provide dataset with labeled for testing.

[Testing Dataset]()
/labshare/VID Dataset/Testing Dataset

There are 9 night testing datasets and 9 daytime testing datasets in VID-S_Dataset floder

```
|--VID-S_Dataset
  |--night_test_n1
    |--drink
    |--normal
    |--talk_left
    |--talk_right
    |--text_left
    |--text_right
  ...
  |--test_n1
    |--drink
    |--normal
    |--talk_left
    |--talk_right
    |--text_left
    |--text_right
  ...
```
VID-D_Dataset folder and VID-M_Dataset folder provide different type of testing image data.

### 2.3 Test
* Go to /V2IA-Net/V2IA-Net_distract/ folder
* Download model weight

[V2IA-Net weight](~) 

Put model weights into /V2IA-Net/V2IA-Net_distract/checkpoints/rgbir_DCL folder

```
15_net_G_A.pth
15_net_G_B.pth
```
* Generate fake image
```
python test.py
```
* Action Accuracy Test
```
python test_acc.py
```
* Binary Accuracy Test
```
python test_dn_acc.py
```
### 3.1 Video Test for distraction detection system
* Go to /V2IA-Net/V2IA-Net_distract/ folder
* Download model weight

[V2IA-Net weight](~) 

Put model weights into /V2IA-Net/V2IA-Net_distract/checkpoints/rgbir_DCL folder
```
15_net_G_A.pth
15_net_G_B.pth
```

* Start test
```
python video_thread.py
```
![distract_test](https://user-images.githubusercontent.com/35215838/181173445-0b10eb90-c9f9-4abf-a19f-1ea94a265728.png)
## Acknowledgments
Our code is developed based on [DCLGAN](https://github.com/JunlinHan/DCLGAN)

In driver distraction detection system, we use [FacePose_pytorch](https://github.com/WIKI2020/FacePose_pytorch) to predict the three-axis angle of the face.
