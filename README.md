# Visible to Infrared Action Net (V2IA-Net)

In order to avoid distracted driving, this paper proposes a **real-time driver distraction monitoring system**. This system is divided into two main spindle, namely driver behavior analysis and head posture analysis. An action classification model is built by deep learning to identify the current driving action, and used to analyze the driver behavior; the angle of the driving head in the three-axis space is estimated by the key feature points. Angle change is evaluated and used to analysis head posture. Finally use the results of the two systems to conduct a comprehensive evaluation to determine whether there is a distraction behavior.

Infrared cameras are rarely used in daily life, there are not enough datasets for development. In this paper, We collect a series of visible and infrared distraction image datasets,**VID Dataset**, and designs a translation models, **V2IA-Net**, using unpaired image to train GAN model that can convert visible images into infrared images. We also add a classification model structure to the generative network, improving the accuracy of the model through partial supervised learning. And at the same time, the model features are used to effectively extract the common features of visible images and infrared images, improve the accuracy of action recognition, and have better performance in driving behavior analysis.

# V2IA-Net Structure
![DCL_m_Generator drawio](https://user-images.githubusercontent.com/35215838/165756609-898e3817-142e-4697-9ea1-422139cb19f6.png)

In this work, our code is developed based on [DCLGAN](https://github.com/JunlinHan/DCLGAN).

We modify the code { models/dcl_model.py, models/networks.py } to define new generator and loss of : 
* class ResnetGenerator()
* class DCLModel()

## System Structure 
![3-16_系統整合e](https://user-images.githubusercontent.com/35215838/165756454-3a97a001-f40d-4e0b-bf6e-c2e67421781c.png)

We use the encoder of generator in V2IA-Net and combine the model with classifier in V2IA-Net to identify driving action categories.

This method can effectively extract common features of visible images and infrared images and improve the accuracy of action recognition.

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
* There are six types of driving distractions : Drink、Normal、Talk left、Talk right、Text left、Text right.
* The dataset contains six men and three women.
* There are two types of datasets, static data and dynamic data.
* Static datasets contain training and testing data, collected while the vehicle is stationary.
* Dynamic datasets contain only testing data, collected while the vehicle is moving.
* Equipment is Garmin Dash Cam Tandem and placed on the upper right of the driver.

### Dataset Download

linker : /labshare/VID Dataset

* video : Original video
* label_for_V2IA-Net : Used for V2IA-Net
* label_for_series_image : Divide data into action sequences (10 frames)

```
|--VID Dataset
  |--video
    |--static.zip
    |--dynamic.zip
  |--label_for_V2IA-Net
    |--rgbir_new_dataset99.zip
    |--test_dataset.zip
    |--dynamic_test_dataset.zip
  |--label_for_series_image
    |--daytime.zip
    |--night.zip
    |--all.zip
    |--series_test_data.zip
```
## Usage 
### Training Set
linker : /labshare/VID Dataset/label_for_V2IA-Net/rgbir_new_dataset99.zip

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
### Train
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
### Testing Set
linker : /labshare/VID Dataset/label_for_V2IA-Net/test_dataset.zip

There are 9 night testing datasets and 9 daytime testing datasets

```
|--dataset
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
### Test
* Go to /V2IA-Net/V2IA-Net_distract/ folder
* Download model weight

[V2IA-Net](~) 

Put model weights into /V2IA-Net/V2IA-Net_distract/checkpoints/rgbir_DCL folder

```
20_net_G_A.pth
20_net_G_B.pth
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
### Video Test
* Go to /V2IA-Net/V2IA-Net_distract/ folder
* Download model weight

[3d resnet50 + lstm](night_best.pth) 

Put model weights into /V2IA-Net/V2IA-Net_distract/ folder
```
night_best.pth
```
[V2IA-Net](~) 

Put model weights into /V2IA-Net/V2IA-Net_distract/checkpoints/rgbir_DCL folder
```
20_net_G_A.pth
20_net_G_B.pth
```
* Start test
```
python video_stream.py
```
![image](https://user-images.githubusercontent.com/35215838/165754906-e5fb88e2-599d-4437-9723-06d219c91b15.png)

## Acknowledgments
Our code is developed based on [DCLGAN](https://github.com/JunlinHan/DCLGAN)

In driver distraction detection system, we use [FacePose_pytorch](https://github.com/WIKI2020/FacePose_pytorch) to predict the three-axis angle of the face.
