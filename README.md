# Visible to Infrared Action Net (V2IA-Net)

# V2IA-Net Structure
![DCL_m_Generator drawio](https://user-images.githubusercontent.com/35215838/165756609-898e3817-142e-4697-9ea1-422139cb19f6.png)

In this work, our code is developed based on [CLGAN](https://github.com/JunlinHan/DCLGAN).
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

[V2IA-Net](~) put in /V2IA-Net/V2IA-Net_distract/checkpoints/rgbir_DCL folder
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

[3d resnet50 + lstm](night_best.pth) put in /V2IA-Net/V2IA-Net_distract/ folder
```
night_best.pth
```
[V2IA-Net](~) put in /V2IA-Net/V2IA-Net_distract/checkpoints/rgbir_DCL folder
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
