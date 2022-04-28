# Visible to Infrared Action Net (V2IA-Net)

## Dependance version
opencv-python 4.4.0.46
## VID Dataset

## Usage 
### Training Set
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
