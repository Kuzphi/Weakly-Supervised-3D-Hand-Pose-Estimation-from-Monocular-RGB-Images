# Weakly-Supervised-3D-Hand-Pose-Estimation-from-Monocular-RGB-Images
Written by Liangjian Chen (Kupzhi@gmail.com)

## Paper Reference:
[ECCV18 weakly-supervised 3D hand](http://openaccess.thecvf.com/content_ECCV_2018/html/Yujun_Cai_Weakly-supervised_3D_Hand_ECCV_2018_paper.html)

## Preprocessing
Download STB dataset from [here](https://sites.google.com/site/zhjw1988/)
Unzip all the file into `data/STB`
Run `STB.py` to get cropped hand 

Download Pre-trained CPM model weight from [here](https://drive.google.com/open?id=1bUIdJi1ofUOqyj8ZGf0kh3gu_RrboYxT) and put it into `./pretrained_weight`

## Training

### Regression
Run `Python train.py --cfg config/train/direct_regression.yaml` to refined the pretrained_weight

### Initial Depth Regularizer
find the best result in the previous training and put the path into the `config/train/depth.yaml` line 70 `PRETRAINED_WEIGHT_PATH`, and 
Run `Python train.py --cfg config/train/direct_regression.yaml` to initialized the weight of depth regularizer 

### End-to-end Training
find the best result in the previous training and put the path into the `config/train/depth.yaml` line 71 and 76 of `PRETRAINED_WEIGHT_PATH`, and Run `Python train.py --cfg config/train/STB.yaml` for the final end-to-end training
