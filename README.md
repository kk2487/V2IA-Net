# Visible to Infrared Action Net (V2IA-Net)

# V2IA-Net Structure
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

linker : /labshare/VID Dataset

* video : Original video
* label_for_V2IA-Net : Used for V2IA-Net

```
|--VID Dataset
  |--video
    |--static.zip
    |--dynamic.zip
  |--label_for_V2IA-Net
    |--rgbir_new_dataset99.zip
    |--test_dataset.zip
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
### Video Test
* Go to /V2IA-Net/V2IA-Net_distract/ folder
* Download model weight

[V2IA-Net](~) 

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
