# Automatic landcover change detection and classification from Satellite images 

## Abstract
This Capstone project is a collaboration between Columbia University at the Data Science Institute and JPMorgan Chase & Co. The objective of this Capstone Project is to build a Deep Learning model to automate the land cover change detection and classification from multitemporal, multiresolution, and multispectral satellite imagery over the Kigali region of Africa. 

The goal of our project is to establish a semantic segmentation model in a specific region of interest describing land cover change between two consecutive time frames (2016 and 2019) using NICFI (Norway's International Climate and Forest Initiative) images. Specifically, we were expected to detect the loss or gain of “tree canopy” land cover class. The scoring metric is the average intersection over union (IoU) between the predicted and ground truth labels. This problem is made difficult by the very different spectral characteristics of the two years which have been derived from the dynamic word labels as well as the lack of high resolution satellite images for training purposes.

The 5m resolution NICFI imagery dataset from 2016 and 2019 (input images) and the Dynamic World Labels (target images) are being used for model training. The scope of the project is to classify the land cover into Tree Canopy, Buildings/Impervious land and Water for each pixel by building a model and then using it to detect land cover change for each class from satellite images of 2016 and 2019. We rely on Dynamic World images for model training purposes, however, we also manually annotate a set of 14 images to ensure that the model performance does not deviate with the real world ground-truth labels. 

We also focus on applying exploratory data analysis strategies to explore our data and find any interesting and meaningful patterns in the images. After researching from multiple resources we decided to train a simple neural network using pixel values as our baseline model for detecting land cover change. We trained a few models using Unet and FCN (Fully Convolutional Network) using patches as well as compressed images for detecting the land cover change. These models have been trained using various pre-trained architectures as their backbone to gain the desired results. The project will include the use of Google Earth Engine (GEE) for the manual extraction of images. We decided to use Google Earth Engine as it combines a multi-petabyte catalog of satellite imagery and geospatial datasets with planetary-scale analysis capabilities. The result of our project is a deep learning model that will enable us to automatically detect the regions affected by deforestation in Kigali in 2019 as compared to 2016.



## Environment setup

Implemented using DFC2021 MSD Baseline
forked from https://github.com/calebrob6/dfc2021-msd-baseline

The following will setup up a conda environment suitable for running the scripts in this repo:
Download this repo and upload the total folder(dfc2021-msd-baseline-master) onto your Google Drive.

Upload Repo_implementation.ipynb (https://github.com/sp3915/dfc2021-msd-baseline/blob/master/Repo_implementation.ipynb) onto colab and run the script with GPUs.

Note: Make sure that the data/image files is accessible. The data/image locations are in csv format in data/splits folder. 

NICFI 2016: https://drive.google.com/drive/folders/1nMKtW5NV9PVwn5o949Gf3vxp0pAvZYmY?usp=sharing
NICFI 2019: https://drive.google.com/drive/folders/10HAtB0LNClpzI7hS2O6Io6pObc29TZ0h?usp=sharing
Dyanmic world 2016: https://drive.google.com/drive/folders/1r7_VfQvOKU_xxZawAlNB9UfAna_CyHlA?usp=sharing
Dyanmic world 2019: https://drive.google.com/drive/folders/1ql9Ypxhh6MAZNC7flpPUQWutnfL8z2Zo?usp=sharing

### `U-Net both` baseline

```
!python train.py --input_fn data/splits/train_list_both.csv --output_dir results/unet_both_baseline_nicfi_main/ --save_most_recent --num_epochs 10

!python inference.py --input_fn data/splits/val_list_both.csv --model_fn results/unet_both_baseline_nicfi_main/most_recent_model.pt --output_dir results/unet_both_baseline_nicfi_main/output/


```

### `U-Net separate` baseline

```
#2016
!python train.py --input_fn data/splits/train_list_2016.csv --output_dir results/unet_2016_baseline_nicfi_main/ --save_most_recent --num_epochs 10

!python inference.py --input_fn data/splits/val_list_2016.csv --model_fn results/unet_2016_baseline_nicfi_main/most_recent_model.pt --output_dir results/unet_2016_baseline_nicfi_main/output/

#2019
!python train.py --input_fn data/splits/train_list_2019.csv --output_dir results/unet_2019_baseline_nicfi_main/ --save_most_recent --num_epochs 10

!python inference.py --input_fn data/splits/val_list_2019.csv --model_fn results/unet_2019_baseline_nicfi_main/most_recent_model.pt --output_dir results/unet_2019_baseline_nicfi_main/output/

```

### `FCN both` baseline

```
!python train.py --input_fn data/splits/train_list_both.csv --output_dir results/fcn_both_baseline_nicfi_main/ --save_most_recent --model fcn --num_epochs 10

!python inference.py --input_fn data/splits/val_list_both.csv --model_fn results/fcn_both_baseline_nicfi_main/most_recent_model.pt --output_dir results/fcn_both_baseline_nicfi_main/output/ --model fcn

```

### `FCN separate` baseline

```
#2016
!python train.py --input_fn data/splits/train_list_2016.csv --output_dir results/fcn_2016_baseline_nicfi_main/ --save_most_recent --model fcn --num_epochs 10

!python inference.py --input_fn data/splits/val_list_2016.csv --model_fn results/fcn_2016_baseline_nicfi_main/most_recent_model.pt --output_dir results/fcn_2016_baseline_nicfi_main/output/ --model fcn

#2019
!python train.py --input_fn data/splits/train_list_both.csv --output_dir results/fcn_2019_baseline_nicfi_main/ --save_most_recent --model fcn --num_epochs 10

!python inference.py --input_fn data/splits/val_list_both.csv --model_fn results/fcn_2019_baseline_nicfi_main/most_recent_model.pt --output_dir results/fcn_2019_baseline_nicfi_main/output/ --model fcn


```
