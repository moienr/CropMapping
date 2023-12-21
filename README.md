# CropMapping of Partially Anotated Time Series Satellite Images with 3D Convolutional Neural Networks



```
Developed for: Iran's Space Research Center's CropMapping competition
Supervised by: Dr. Sara Attarchi
Authors: Mohammad Ramezanpour, Moien Rangzan
```


The goal of the competition was to predict the crop type of a given time series of satellite images at least one month before the harvest. 

# Introduction

## Model
Our model is Siamese 3D segmentaion network that takes as input two timeseries of Sentinel-1 and Sentinel-2 images, fuses them and outputs a segmentation mask. 


## Training procedure
Since the dataset provided by ISRC is partially anotated, meaning that in each image there might be crops that are not anotated, we used transfer learning to train our model. 

- **Step 1**: First we downloaded a dataset of around 150GB of data using Google Earth Engine(GEE), consisting of 6 months of Sentinel-1 and Sentinel-2 images, and their corresponding anotations from [EUCROPMAP 2018 Dataset](https://developers.google.com/earth-engine/datasets/catalog/JRC_D5_EUCROPMAP_V1) freely available on GEE.
- **Step 2**: We trained our model on this dataset for 11 epochs.

- **Step 3**: Then we changed the segmentation head of our model and trained it on the ISRC dataset for 5 epochs. With a lower learning rate of `1e-5` for trained layers and `1e-4` for new layers.

This procedure enables the model to learn the features of the ISRC dataset while not forgetting the features of the EUCROPMAP dataset, and the low epoch number for the ISRC dataset prevents the model from overfitting on partially anotated data.


# How to use

## Dataset
A through explaination of how build a dataset from gournd truth shapefiles is provided in [main_dataset/README.md](./main_dataset/README.md)

## Evaluation
After building the dataset, you can evaluate the model on the dataset using the following command:

```bash
> conda activate pytorch_env
> python eval.py --eval_crop <crop_to_be_evaluated> --dataset_dir_path <path to dataset> --trained_model_path <path to trained model>
```

