import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from glob import glob
from skimage import io
import os
from torchvision import datasets, transforms
import matplotlib
import os
import gc
import random
from datetime import date, datetime
import json
import pprint
from model.model import DualUNet3D, IranCropModel
from train_utils import train, binary_mask_accuracy, calculate_dataset_metrics
from config import DEVICE, CROP_LIST, DATASET_DIRS
from plot import plot_output_crop_map, plot_s2_img
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import random_split
import dataset.data_loaders as dl
from plot import plot_train_test_losses
import argparse

parser = argparse.ArgumentParser(description='Finetune script')
parser.add_argument('--augmentation', '-a', action='store_true', help='Enable data augmentation')
parser.add_argument('--batch_size', '-b', type=int, default=16, help='Batch size')
parser.add_argument('--eu_model_state_dict', '-e', type=str, default='./model/StateDict_epoch11_EU.pth', help='Path to the pretrained EU model state dictionary')
parser.add_argument('--num_epoch', '-n', type=int, default=10, help='Number of epochs')
parser.add_argument('--save_model_dir', '-s', type=str, default='./model/saved_models/', help='Directory to save the models state dictionaries')
parser.add_argument('--save_sampler_file_path', '-sf', type=str, default='./results/samplers/', help='Path to save the sampler files')
parser.add_argument('--load_sampler_file_path', '-lf', type=str, default=None, help='Path to load the sampler files')

args = parser.parse_args()

AUGMENTATION = args.augmentation
BATCH_SIZE = args.batch_size
EU_MODEL_STATE_DICT = args.eu_model_state_dict
NUM_EPOCH = args.num_epoch
SAVE_MODEL_DIR = args.save_model_dir
SAVE_SAMPLER_FILE_PATH = args.save_sampler_file_path
LOAD_SAMPLER_FILE_PATH = args.load_sampler_file_path

def main():
    s1_transform = transforms.Compose([dl.NormalizeS1(),dl.myToTensor(dtype=torch.float32)])
    s2_transform = transforms.Compose([dl.NormalizeS2(),dl.myToTensor(dtype=torch.float32)])
    augmentation = dl.Augmentations() if AUGMENTATION else None


    if len(DATASET_DIRS) != len(CROP_LIST):
        raise Exception("Number of directories and crops should be the same.")

    s1_dirs = [dir + "s1/" if dir else "" for dir in DATASET_DIRS]
    s2_dirs = [dir + "s2/" if dir else "" for dir in DATASET_DIRS]
    crop_map_dirs = [dir + "crop_map/" if dir else "" for dir in DATASET_DIRS]

    train_dataset_list = []
    valid_dataset_list = []
    for i ,dirs in enumerate(zip(s1_dirs,s2_dirs,crop_map_dirs)):
        # if there is only one crop in the dataset, outpus is a binary mask, otherwise it is a one-hot encoded mask.
        crop_map_transform = transforms.Compose([
                dl.CropMapTransformIran(crop_type=CROP_LIST[i]),
                dl.myToTensor(dtype=torch.float32)
                                                ])
        if dirs[0] == "":
            continue
        dataset = dl.Sen12Dataset(s1_dir=dirs[0],
                                s2_dir=dirs[1],
                                crop_map_dir=dirs[2],
                                s1_transform=s1_transform,
                                s2_transform=s2_transform,
                                crop_map_transform=crop_map_transform,
                                augmentation=augmentation,
                                verbose=False)
        dataset_tarin_split, dataset_valid_split = random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
        train_dataset_list.append(dataset_tarin_split)
        valid_dataset_list.append(dataset_valid_split)
        
    if len(train_dataset_list) == 1:
        train_dataset = train_dataset_list[0]
        valid_dataset = valid_dataset_list[0]
    else:
        train_dataset = ConcatDataset(train_dataset_list)
        valid_dataset = ConcatDataset(valid_dataset_list)

    print(f"Train, Valid Dataset Size Before Sampling: {len(train_dataset)}, {len(valid_dataset)}")
    train_sampler = dl.BalancedSampler(train_dataset, ratio=0.00, shuffle=True,
                                       save_indices_file= SAVE_SAMPLER_FILE_PATH + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "_train_indices.pkl",
                                       load_indices_file=LOAD_SAMPLER_FILE_PATH)
    test_sampler = dl.BalancedSampler(valid_dataset, ratio=0.00, shuffle=True,
                                       save_indices_file= SAVE_SAMPLER_FILE_PATH + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + "_test_indices.pkl",
                                       load_indices_file=LOAD_SAMPLER_FILE_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=2, sampler=test_sampler)    
    print(f"Train, Valid Dataset Size After Sampling: {len(train_loader)*BATCH_SIZE}, {len(valid_loader)*BATCH_SIZE}")
    # Load the model
    dualunet3d_dict = torch.load(EU_MODEL_STATE_DICT)
    dualunet3d = DualUNet3D(s1_in_channels=2, s2_in_channels=10, out_channels=21,ts_depth=6,non_lin='sigmoid').to(DEVICE)
    dualunet3d.load_state_dict(dualunet3d_dict)
    # Change segmentation head to have 8 output channels (number of crops in the dataset)
    model = IranCropModel(out_channels=len(CROP_LIST), dualunet3d=dualunet3d, non_lin=nn.Sigmoid()).to(DEVICE)
    
    # Example usage
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    scheduler_type = "constant"  # Options: "constant", "step", "plateau". Choose which scheduler to use.
    scheduler_kwargs = {}


    # Get the parameters of the final_conv layer
    final_conv_params = model.final_conv.parameters()

    # Get the parameters of all other layers
    other_params = [param for name, param in model.named_parameters() if 'final_conv' not in name]

    # Create the optimizer
    optimizer = torch.optim.Adam([
        {'params': final_conv_params, 'lr': 1e-4},
        {'params': other_params, 'lr': 1e-5}
    ])          
    
    results = train(model, train_loader, valid_loader, criterion, optimizer, scheduler_type, NUM_EPOCH, **scheduler_kwargs)
    
    # Create the directory if it doesn't exist
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)


    torch.save(model.state_dict(), f"{SAVE_MODEL_DIR}StateDict_epoch{NUM_EPOCH}_Full.pth")
    print("Model saved successfully.")
    
    train_loss_history = np.array(results["train_loss_history"]).reshape(1, -1)
    val_loss_history = np.array(results["val_loss_history"]).reshape(1, -1) 
    plot_train_test_losses(train_loss_history, val_loss_history)
    
    
if __name__ == "__main__":
    main()