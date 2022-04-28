# Visible to Infrared Action Net (V2IA-Net)

# V2IA-Net Structure
![DCL_m_Generator drawio](https://user-images.githubusercontent.com/35215838/165700320-a0b6fd80-f365-482f-83d7-30ec001bb91e.png)

In this work, our code is developed based on [CLGAN](https://github.com/JunlinHan/DCLGAN).
We modify the code { models/dcl_model.py, models/networks.py } to define new generator and loss of : 
* class ResnetGenerator()
* class DCLModel()

## System Structure 
![3-16_系統整合e](https://user-images.githubusercontent.com/35215838/165701268-fe84e2b6-601d-4bdb-bc31-475717781aeb.png)

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
### Training
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
## Acknowledgments
Our code is developed based on [CLGAN](https://github.com/JunlinHan/DCLGAN)
