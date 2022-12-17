# Automatic landcover change detection and classification from Satellite images 

Implemenataion using DFC2021 MSD Baseline
Forked from https://github.com/calebrob6/dfc2021-msd-baseline


## Environment setup

The following will setup up a conda environment suitable for running the scripts in this repo:
Download this repo and upload the total folder(dfc2021-msd-baseline-master) onto your Google Drive.

Upload Repo_implementation.ipynb onto colab and run the script with GPUs.

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
