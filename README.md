# CropMapping of Partially Annotated Time Series Satellite Images with 3D Convolutional Neural Networks



```
Developed for Iran's Space Research Center's CropMapping competition
Supervised by: Dr. Sara Attarchi
Authors: Mohammad Ramezanpour, Moien Rangzan
```


The goal of the competition was to predict the crop type of a given time series of satellite images at least one month before the harvest. 

# Introduction

## Model
Our model is a Siamese 3D segmentation network that takes as input two time-series of Sentinel-1 and Sentinel-2 images, fuses them, and outputs a segmentation mask. 


## Training procedure
Since the dataset provided by ISRC is partially annotated, meaning that in each image, there might be un-annotated crops, we used transfer learning to train our model. 

- **Step 1**: First, we downloaded a dataset of around 150GB of data using Google Earth Engine(GEE), consisting of 6 months of Sentinel-1 and Sentinel-2 images and their corresponding annotations from [EUCROPMAP 2018 Dataset](https://developers.google.com/earth-engine/datasets/catalog/JRC_D5_EUCROPMAP_V1) freely available on GEE.
- **Step 2**: We trained our model on this dataset for 11 epochs.

- **Step 3**: Then we changed the segmentation head of our model and trained it on the ISRC dataset for 5 epochs, with a lower learning rate of `1e-5` for trained layers and `1e-4` for new layers.

This procedure enables the model to learn the features of the ISRC dataset while not forgetting the features of the EUCROPMAP dataset, and the low epoch number for the ISRC dataset prevents the model from overfitting on partially annotated data.


# How to use

**Clone the Repository**:
   ```bash
   git clone https://github.com/moienr/CropMapping.git
   ```

**Create the Environments**:

    ```bash
    conda env create -f environment.yml
    conda env create -f pytorch_env.yml
    ```




## Dataset

A thorough explanation of how to build a dataset from ground truth shapefiles is provided in [main_dataset/README.md](./main_dataset/README.md)

Then the new dataset ["main_dataset/output/Iran_ROI.xlsx"](./main_dataset/output/Iran_ROI.xlsx) can be downloaded using [Dataset Generator Notebook](./dataset/iran_ds_generator.ipynb).


## Evaluation
After building the dataset, you can evaluate the model on the dataset using the following command:

```bash
> conda activate pytorch_env
> python eval.py --eval_crop <crop_to_be_evaluated> --dataset_dir_path <path to dataset> --trained_model_path <path to trained model> -th <threshold>
```


