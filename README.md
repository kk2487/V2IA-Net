# Visible to Infrared Action Net (V2IA-Net)

## Dependance version
opencv-python 4.4.0.46
## VID Dataset
linker : /labshare/VID Dataset

```
|--VID Dataset
  |--Video
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
