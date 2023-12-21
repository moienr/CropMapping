import torch
import torch.optim as optim
import os
from torchvision import datasets, transforms
import matplotlib
import gc
import random
from datetime import date, datetime
import json
import pprint
from model.model import DualUNet3D, IranCropModel
from train_utils import train, binary_mask_accuracy, calculate_dataset_metrics
from config import DEVICE, CROP_LIST
from plot import plot_output_crop_map, plot_s2_img
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import random_split
import dataset.data_loaders as dl
from plot import plot_train_test_losses
import argparse




parser = argparse.ArgumentParser()
parser.add_argument("--eval_crop", "-ec", type=str, default="tomato", help="Crop type for evaluation")
parser.add_argument("--dir_path", "-dp", type=str, default="./crop_map_dataset_Iran_tomato/"
                    , help="Directory path of Crop Dataset")
parser.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size (keep the same as training)")
parser.add_argument("--save_sampler_file", "-ss", type=bool, default=True, help="Save sampler file or not")
parser.add_argument("--load_sampler_file_from_path", "-lp", type=str, default=None, help="Path to load sampler file")
parser.add_argument("--trained_model_path", "-tp", type=str, default="./model/StateDict_epoch11_EU.pth", help="Path to trained model")
parser.add_argument("--threshold", "-th", type=float, default=0.35, help="Threshold value")
args = parser.parse_args()

EVAL_CROP = args.eval_crop
DIR_PATH = args.dir_path
BATCH_SIZE = args.batch_size
SAVE_SAMPLER_FILE = args.save_sampler_file
SAVE_SAMPLER_FILE_PATH = f"./results/samplers/{EVAL_CROP}_sampler_indices.pkl" if SAVE_SAMPLER_FILE else None


LOAD_SAMPLER_FILE_PATH = args.load_sampler_file_from_path
TRAIED_MODEL_PATH = args.trained_model_path
THRESHOLD = args.threshold
SAVE_RESULTS_PATH = f"./model/results/results_{EVAL_CROP}_Threshold_{THRESHOLD}_{date.today().strftime('%d-%m-%Y')}.json"

os.makedirs(os.path.dirname(SAVE_RESULTS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SAVE_SAMPLER_FILE_PATH), exist_ok=True)


def main():

    dir = DIR_PATH
    s1_dir = dir + "s1/" 
    s2_dir = dir + "s2/" 
    crop_map_dir = dir + "crop_map/"


    s1_transform = transforms.Compose([dl.NormalizeS1(),dl.myToTensor(dtype=torch.float32)])
    s2_transform = transforms.Compose([dl.NormalizeS2(),dl.myToTensor(dtype=torch.float32)])
    crop_map_transform = transforms.Compose([
            dl.CropMapTransformIran(crop_type=EVAL_CROP),
            dl.myToTensor(dtype=torch.float32)
                                            ])

    test_dataset = dl.Sen12Dataset(s1_dir=s1_dir,
                            s2_dir=s2_dir,
                            crop_map_dir=crop_map_dir,
                            s1_transform=s1_transform,
                            s2_transform=s2_transform,
                            crop_map_transform=crop_map_transform,
                            augmentation=None,
                            verbose=False)

    test_sampler = dl.BalancedSampler(test_dataset, ratio=0.00, shuffle=True, save_indices_file=SAVE_SAMPLER_FILE_PATH, load_indices_file=LOAD_SAMPLER_FILE_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2, sampler=test_sampler)
    
    dualunet3d_dict = torch.load(TRAIED_MODEL_PATH)
    dualunet3d = DualUNet3D(s1_in_channels=2, s2_in_channels=10, out_channels=21,ts_depth=6,non_lin='sigmoid').to(DEVICE)
    dualunet3d.load_state_dict(dualunet3d_dict)
    model = IranCropModel(out_channels=len(CROP_LIST), dualunet3d=dualunet3d, non_lin=nn.Sigmoid()).to(DEVICE)
    
    
    eval_results = calculate_dataset_metrics({f"{EVAL_CROP.capitalize()}_Test": test_loader}, model, threshold=THRESHOLD, channel=CROP_LIST.index(EVAL_CROP))
    pprint.pprint(eval_results)
    # Save the results dictionary as a JSON file
    with open(SAVE_RESULTS_PATH, "w") as file:
        json.dump(eval_results, file)
    print(f"Saved results to {SAVE_RESULTS_PATH}")

if __name__ == "__main__":
    main()
    
    