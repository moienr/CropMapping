import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from skimage import io
import os
from torchvision import datasets, transforms
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
import random
# from changedetection.utils import get_binary_change_map

def get_all_files(path:str, file_type=None)->list:
    """Returns all the files in the specified directory and its subdirectories.
    
    e.g 2021/s1_imgs/ will return all the files in `2021/s1_imgs/` subfolders which are `train` and `test`
    
    it will return the names like `train/2021_01_01.tif` and `test/2021_01_01.tif` if subfolders are present
    if not it will return the names like `2021_01_01.tif`
    """
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file_type is None:
                file_list.append(os.path.relpath(os.path.join(root, file), path))
            elif file.endswith(file_type):
                file_list.append(os.path.relpath(os.path.join(root, file), path))
            
    return file_list




def find_difference(list1, list2):
    """Find the difference between two lists."""
    # Use list comprehension to find elements in list1 that are not in list2
    difference1 = [item for item in list1 if item not in list2]
    
    # Use list comprehension to find elements in list2 that are not in list1
    difference2 = [item for item in list2 if item not in list1]
    
    # Concatenate the two differences to get the complete list of different elements
    result = difference1 + difference2
    
    return result

def mask_fill_percentage(path, image_shape=(64,64))->list:
    """ Returns the percentage of pixels that are not zero in the mask images in the specified directory.
    """
    image_files = [file for file in os.listdir(path) if file.endswith('.tif')]
    # print(f"len of image files: {len(image_files)}")
    pixel_sums = []
    for image_file in image_files:
        image_path = os.path.join(path, image_file)
        image = io.imread(image_path)
        pixel_sum = image.sum()
        pixel_sums.append(pixel_sum)
    pixel_avgs = [(pixel_sum*100) / (image_shape[0] * image_shape[1]) for pixel_sum in pixel_sums]
    return pixel_avgs


def find_empty_masks(path:str, file_type=".tif")->list:
    """returns the names of the empty masks in the specified directory"""
    empty_masks = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(file_type):
                file_path = os.path.join(root, file)
                image = io.imread(file_path)
                if image.sum() == 0:
                    empty_masks.append(file_path)
    return empty_masks


class Sen12Dataset(Dataset):
    """Dataset class for the Sen12MS dataset."""
    def __init__(self,
               s1_dir,
               s2_dir,
               crop_map_dir,
               s2_bands: list = None ,
               s1_transform = None,
               s2_transform = None,
               crop_map_transform = None,
               augmentation = None,
               verbose=False):
        """
        Initialize the Sen12Dataset.

        Args:
            s1_dir (str): Directory path for S1 images.
            s2_dir (str): Directory path for S2 images.
            crop_map_dir (str): Directory path for crop map images.
            s2_bands (list, optional): List of indices of S2 bands to use. Defaults to None.
            s1_transform (callable, optional): Transform to apply on S1 images. Defaults to None.
            s2_transform (callable, optional): Transform to apply on S2 images. Defaults to None.
            crop_map_transform (callable, optional): Transform to apply on crop map images. Defaults to None.
            augmentation (callable, optional): Augmentation function to apply on images. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        self.verbose = verbose
        # Set the directories for the four sets of images
        self.s1_dir = s1_dir
        self.s2_dir = s2_dir
        self.crop_map_dir = crop_map_dir
        self.augmentation = augmentation
        
        # Get the names of the S2 and S1 time-2 images and sort them
        self.s1_dates = os.listdir(s1_dir)
        self.s1_dates.sort()
        self.num_dates = len(self.s1_dates)
        self.s2_dates = os.listdir(s2_dir)
        self.s2_dates.sort()
        self.file_names= get_all_files(self.s1_dir + "//" + self.s1_dates[0])
        self.file_names.sort()

        assertion_names = get_all_files(self.s2_dir + "//" + self.s2_dates[3])
        assertion_names.sort()
        crop_map_names = get_all_files(crop_map_dir)
        crop_map_names.sort()
        
        
        # Verify that the four sets of images have the same names
        if self.s1_dates != self.s2_dates:
            diff = find_difference(self.s1_dates, self.s2_dates)
            raise ValueError(f"S1 and S2 directories do not contain the same dates | Diff Len: {len(diff)} | Diffrencce: {diff}")
        if self.file_names != assertion_names:
            diff = find_difference(self.file_names, assertion_names)
            raise ValueError(f"S2 date directories do not contain the same image pairs | Diff Len: {len(diff)} | Diffrencce: {diff}")
        if self.file_names != crop_map_names:
            diff = find_difference(self.file_names, crop_map_names)
            raise ValueError(f"S1 and S2 directories do not contain the same image pairs as Cropmap | Diff Len: {len(diff)} | Diffrencce: {diff}")
        
        
        self.s2_bands = s2_bands if s2_bands else None 

        self.s1_transform = s1_transform
        self.s2_transform = s2_transform
        self.crop_map_transform = crop_map_transform


    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.file_names)

    def __getitem__(self, index):
        """
        Get the item at the given index.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: A tuple containing the S1 image, S2 image, and crop map. Each a Tensor with shape (channels, height, width).
        """
        img_name = self.file_names[index] 

        if self.verbose: print(f"Image name: {img_name}")  
        
        s1_images = []
        s2_images = []
        
        for d in self.s1_dates:
            s1_img_path = os.path.join(self.s1_dir,d,img_name)
            s1_img = io.imread(s1_img_path)
            s1_images.append(s1_img)
            s2_img_path = os.path.join(self.s2_dir,d,img_name)
            s2_img = io.imread(s2_img_path)
            if self.s2_bands: s2_img = s2_img[self.s2_bands,:,:]
            s2_images.append(s2_img)
            
        crop_map_path = os.path.join(self.crop_map_dir,img_name)
        crop_map = io.imread(crop_map_path)
        if self.verbose: print(f'crop_map shape apon reading: {crop_map.shape}')
            
        if self.verbose: print(f's2 shape apon reading: {s2_img.shape}')
        if self.verbose: print(f's1 shape apon reading: {s1_img.shape}')
        
        if self.s1_transform:
            s1_images = [self.s1_transform(s1_img) for s1_img in s1_images]
        if self.s2_transform:
            s2_images = [self.s2_transform(s2_img) for s2_img in s2_images]
        if self.crop_map_transform:
            crop_map = self.crop_map_transform(crop_map)
        if self.verbose: print(f'crop_map shape apon transform: {crop_map.shape}')
               
        if self.augmentation:
             s1_images, s2_images, crop_map = self.augmentation(s1_images, s2_images, crop_map)
        
        # Stack images for 3D Convolution to shape (channels, depth, height, width)
        s1_img = torch.stack(s1_images , dim=1)
        s2_img = torch.stack(s2_images , dim=1)
        
        if self.verbose: print(f's2 shape apon stacking: {s2_img.shape}')
        if self.verbose: print(f's1 shape apon stacking: {s1_img.shape}')
        
        




        if self.verbose:
            print(f"stacked s1_img shape: {s1_img.shape}")
            print(f"stacked s2_img shape: {s2_img.shape}")
            print(f"crop_map shape: {crop_map.shape}")
        
        check_tensor_values([s2_img, s1_img],
                            ["s2_img", "s1_img"])
        
        

        return s1_img, s2_img, crop_map

def check_tensor_values(tensor_list, input_names):
    for i, tensor in enumerate(tensor_list):
        if torch.any(tensor > 1) or torch.any(tensor < 0):
            input_name = input_names[i]
            raise ValueError(f"Values of {input_name} tensor must be between 0 and 1.")
        
#########################################################################################################################################
###################################################### Transfroms #######################################################################
#########################################################################################################################################      
      
        
class myToTensor:
    """Transform a pair of numpy arrays to PyTorch tensors"""
    def __init__(self,dtype=torch.float32):
        """Transform a pair of numpy arrays to PyTorch tensors
        
        Args
        ---
            `dtype` (torch.dtype): Data type for the output tensor (default: torch.float32)
        """
        self.dtype = dtype
        
    def reshape_tensor(self,tensor):
        """Reshape a 2D or 3D tensor to the expected shape of pytorch models which is (channels, height, width)
        
        Args
        ---
            `tensor`(numpy.ndarray): Input tensor to be reshaped
        
        Returns:
            torch.Tensor: Reshaped tensor
        """
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3 and tensor.shape[2] < tensor.shape[0]:
            tensor = tensor.permute((2,0,1)) # Channels first
        elif tensor.dim() == 3 and tensor.shape[2] > tensor.shape[0]:
            pass
        else:
            raise ValueError(f"Input tensor shape is unvalid: {tensor.shape}")
        return tensor

    def __call__(self,sample):
        return self.reshape_tensor(torch.from_numpy(sample)).to(dtype=self.dtype)

class NormalizeS1:
    """
    Class for normalizing Sentinel-1 images between 0 and 1 for use with a pix2pix model.
    """

    def __init__(self, s1_min=-25, s1_max=10, check_nan=True, fix_nan=False):
        """
        Args
        ---
            `s1_min` (float): Minimum value for Sentinel-1 data. Default is -25.
            `s1_max` (float): Maximum value for Sentinel-1 data. Default is 10.
            `check_nan` (bool): Check for NaN values in the images, if there is it will rasie an error. Default is False.
            `fix_nan` (bool): Check for NaN values in the images, if there is it will replace it with `0.01`. Default is False.
        """
        self.s1_min = s1_min
        self.s1_max = s1_max
        self.check_nan = check_nan
        self.fix_nan = fix_nan

    def __call__(self, s1_img):
        """
        Normalize Sentinel-1 images for use with a pix2pix model.

        Args:
            s1_img (numpy.ndarray): Sentinel-1 image as a numpy array.

        Returns:
            numpy.ndarray: Normalized Sentinel-1 image. Between 0 and 1
        """
        # Sentinel 1 VV image  is between -25 and 10 dB (we insured that in the data preparation step)
        # print(np.min(target),np.max(target))
        s1_img[s1_img > self.s1_max] = self.s1_max
        s1_img[s1_img < self.s1_min] = self.s1_min

        s1_img = (s1_img - np.min(s1_img)) / (np.max(s1_img) - np.min(s1_img))
        s1_img[s1_img >= 1] = 1 - 0.0001
        s1_img[s1_img <= 0] = 0 + 0.0001

        if self.check_nan:
            if np.isnan(s1_img).any():
                raise ValueError("s1_img contains NaN values")
        elif self.fix_nan:
            if np.isnan(s1_img).any():
                s1_img[np.isnan(s1_img)] = 0.01

        return s1_img


class NormalizeS2:
    """
    Class for normalizing Sentinel-2 images between 0 and 1 for use with a pix2pix model.
    """

    def __init__(self, s2_min=0, s2_max=1, check_nan=True, fix_nan=False):
        """
        Args
        ---
            `s2_min` (float): Minimum value for Sentinel-2 data. Default is 0.
            `s2_max` (float): Maximum value for Sentinel-2 data. Default is 1.
            `check_nan` (bool): Check for NaN values in the images, if there is it will rasie an error. Default is False.
            `fix_nan` (bool): Check for NaN values in the images, if there is it will replace it with `0.01`. Default is False.
        """
        self.s2_min = s2_min
        self.s2_max = s2_max
        self.check_nan = check_nan
        self.fix_nan = fix_nan

    def __call__(self, s2_img):
        """
        Normalize Sentinel-2 images for use with a pix2pix model.

        Args:
            s2_img (numpy.ndarray): Sentinel-2 image as a numpy array.

        Returns:
            numpy.ndarray: Normalized Sentinel-2 image.
        """
        # Sentinel 2 image  is between 0 and 1 it is surface reflectance so it can't be more than 1 or less than 0
        s2_img[s2_img >= self.s2_max] = self.s2_max - 0.0001
        s2_img[s2_img <= self.s2_min] = self.s2_min + 0.0001



        if self.check_nan:
            if np.isnan(s2_img).any():
                raise ValueError("s2_img contains NaN values")
        elif self.fix_nan:
            if np.isnan(s2_img).any():
                s2_img[np.isnan(s2_img)] = 0.01

        return s2_img


class S2S1Normalize:
    """
    Class for normalizing Sentinel-2 and Sentinel-1 images for use with a pix2pix model.
    """

    def __init__(self, s1_min=-25, s1_max=10, s2_min=0, s2_max=1, check_nan=False, fix_nan=False):
        """
        Args
        ---
            `s1_min` (float): Minimum value for Sentinel-1 data. Default is -25.
            `s1_max` (float): Maximum value for Sentinel-1 data. Default is 10.
            `s2_min` (float): Minimum value for Sentinel-2 data. Default is 0.
            `s2_max` (float): Maximum value for Sentinel-2 data. Default is 1.
            `check_nan` (bool): Check for NaN values in the images, if there is it will rasie an error. Default is False.
            `fix_nan` (bool): Check for NaN values in the images, if there is it will replace it with `0.01`. Default is False.
        """
        self.normalize_s1 = NormalizeS1(s1_min=s1_min, s1_max=s1_max, check_nan=check_nan, fix_nan=fix_nan)
        self.normalize_s2 = NormalizeS2(s2_min=s2_min, s2_max=s2_max, check_nan=check_nan, fix_nan=fix_nan)

    def __call__(self, sample):
        """
        Normalize Sentinel-2 and Sentinel-1 images for use with a pix2pix model.

        Args:
            sample (tuple): Tuple containing Sentinel-2 and Sentinel-1 images as numpy arrays.

        Returns:
            tuple: Tuple containing normalized Sentinel-2 and Sentinel-1 images.
        """
        s2_img, s1_img = sample
        s2_img = self.normalize_s2(s2_img)
        s1_img = self.normalize_s1(s1_img)

        return s2_img, s1_img

import numpy as np


class CropMapTransform:
    def __init__(self, crop_values=[211, 212, 213, 214, 215, 216, 217,
                              218, 219, 221, 222, 223, 230, 231,
                              232, 233, 240, 250, 290, 300, 500]
                 ):
        self.crop_values = crop_values

    def __call__(self, crop_map):
        # Convert crop map to binary bands
        binary_bands = []
        crop_map = crop_map.squeeze()
        # Iterate through each crop value
        for crop in self.crop_values:
            # Create a binary band where crop_map equals crop value 
            binary_band = np.where(crop_map == crop, 1, 0)
            # Append the binary band to the list
            binary_bands.append(binary_band)

        # Stack bands along the first axis
        result = np.stack(binary_bands, axis=0)

        return result

# Example usage:
# transform = CropMapTransform()
# crop_map = np.random.randint(low=211, high=500, size=(1, 64, 64), dtype=np.int16)
# result = transform(crop_map)

class CropMapTransformIran:
    def __init__(self, crop_type=None, crops_list=["canola", "cotton", "lentils", "maize",
                                                   "onion", "pea", "sugarbeet", "tomato"]
                 ):
        """
        Initializes an instance of CropMapTransformIran.
        
        Args:
        ---
            crop_type (str, optional): The type of crop. Defaults to None. (If None, it returns a binary map. Else, it returns a multi-channel map.)
            crops_list (list, optional): The list of crop types. Defaults to ["canola", "cotton", "lentils", "maize",
                                       "onion", "pea", "sugarbeet", "tomato"].
        """
        self.crop_type = crop_type
        self.crops_list = crops_list
        if crop_type:
            if crop_type not in crops_list:
                raise ValueError(f"crop_type must be one of {crops_list}")
            self.crop_index = crops_list.index(crop_type)

    def __call__(self, crop_map):
        """
        Applies the crop map transformation.

        Args:
        ---
            crop_map (numpy.ndarray): The input crop map.

        Returns:
        ---
            numpy.ndarray: The transformed crop map.
                If crop_type is None, it returns a binary single-channel map.
                Otherwise, it returns a multi-channel map where the index in the list is the map and others are zero.
        """
        crop_map[crop_map < 0.01] = 0
        crop_map[crop_map > 0.01] = 1

        if self.crop_type:
            crop_map = crop_map.squeeze()
            crop_map_copy = crop_map.copy()
            crop_map = np.zeros((len(self.crops_list), crop_map.shape[0], crop_map.shape[1]))
            crop_map[self.crop_index] = crop_map_copy

        return crop_map

   

import torch
from torchvision import transforms
from PIL import Image

class Augmentations:
    """
    Data Augmentation Class
    This class applies the same augmentation to three images simultaneously.
    The images are padded, randomly flipped, rotated, and then cropped.
    
    Warning: When using the Dataset, make sure to call it only once for each image. For instance:
    ```
    idx = 2
    s1s2 = s1s2_dataset[idx]
    img1 = s1s2[0][1,3,:,:]
    img2 = s1s2[1][1,3,:,:]
    img3 = s1s2[2][0,:,:]
    ```
    
    but if you use it like this:
    ```
    idx = 2
    img1 = s1s2_dataset[idx][0][1,3,:,:]
    img2 = s1s2_dataset[idx][1][1,3,:,:]
    img3 = s1s2_dataset[idx][2][0,:,:]
    ```
    you will get 3 images with 3 different augmentations.
    
    """
    def __init__(self, aug_prob=0.5, out_shape=(64, 64)):
        self.aug_prob = aug_prob
        self.padding = out_shape[0] // 4
        self.out_shape = out_shape
        self.transforms = transforms.Compose([
            transforms.Pad((self.padding, self.padding), padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=aug_prob),
            transforms.RandomVerticalFlip(p=aug_prob),
            transforms.RandomRotation((0,90)),
            transforms.CenterCrop(size=self.out_shape)
        ])

    def __call__(self, s1_img_list, s2_img_list, mask):
        # Generate a random seed based on which the transformations will be applied
        seed = int(torch.randint(1,1000, size=(1,)).item())
        # print(f"seed: {int(seed)}")
        # Apply the same transformations to each image/mask based on the generated seed
        def apply_transform(seed, img):
            torch.manual_seed(seed)
            random.seed(seed) 
            return self.transforms(img)
        s1_img_list = [apply_transform(seed, img) for img in s1_img_list]
        s2_img_list = [apply_transform(seed, img) for img in s2_img_list]
        mask_aug = apply_transform(seed, mask)

        return s1_img_list, s2_img_list, mask_aug

class BalancedSampler(torch.utils.data.sampler.Sampler):
    """ Samples only `ratio` of the data from empty masks (majority class) and all the data from non-empty masks (minority class)"""
    def __init__(self, dataset, ratio=0.1, shuffle = False, verbose= True):
        """ Samples only `ratio` of the data from empty masks (majority class) and all the data from non-empty masks (minority class)
        Input:
        ---
            dataset (Sen12Dataset): The dataset to be sampled.
            ratio (float): The ratio of the majority class to be sampled. Default is 0.1.
        """
        self.verbose = verbose
        self.shuffle = shuffle
        self.dataset = dataset
        self.ratio = ratio
        self.empty_mask_indices, self.non_empty_mask_indices = self.get_empty_and_nonempty_mask_indices()
        self.num_empty_mask_indices = len(self.empty_mask_indices)
        self.num_non_empty_mask_indices = len(self.non_empty_mask_indices)
        self.num_samples = int(self.num_empty_mask_indices * self.ratio) + self.num_non_empty_mask_indices
        
    def get_empty_and_nonempty_mask_indices(self):
        empty_mask_indices = []
        non_empty_mask_indices = []
        for i in range(len(self.dataset)):
            self.verbose and print(f"Getting empty and non-empty mask indices: {i}/{len(self.dataset)}", end="\r")
            crop_map = self.dataset[i][2]
            if crop_map.sum() == 0:
                empty_mask_indices.append(i)
            else :
                non_empty_mask_indices.append(i)
            
        return empty_mask_indices, non_empty_mask_indices

    
    def __iter__(self):
        empty_mask_indices = random.sample(self.empty_mask_indices, k=int(self.num_empty_mask_indices * self.ratio))
        indices = empty_mask_indices + self.non_empty_mask_indices
        self.shuffle and random.shuffle(indices) # shuffle indices if shuffle is True
        return iter(indices)
    
    def __len__(self):
        return self.num_samples




# from torchvision.transforms import functional as F

# class Augmentations:
#     def __init__(self, out_shape=(64, 64)):
#         self.padding = out_shape[0] // 4
#         self.out_shape = out_shape

#     def __call__(self, s1_img_list, s2_img_list, mask):
#         # seed = int(torch.randint(1, 1000, size=(1,)).item())
#         flip_horizontal = bool(torch.randint(0, 2, size=(1,)).item())
#         flip_vertical = bool(torch.randint(0, 2, size=(1,)).item())
#         rotation_angle = int(torch.randint(0, 91, size=(1,)).item())

#         def apply_transform(img, flip_horizontal, flip_vertical, rotation_angle):
#             # torch.manual_seed(seed)
#             # random.seed(seed)
#             img = F.pad(img, (self.padding, self.padding), padding_mode='reflect')
#             if flip_horizontal:
#                 img = F.hflip(img)
#             if flip_vertical: 
#                 img = F.vflip(img)
#             img = F.rotate(img, rotation_angle)
#             img = F.center_crop(img, self.out_shape)
#             return img
        
#         apply_t = lambda img: apply_transform(img, flip_horizontal, flip_vertical, rotation_angle)

#         s1_img_list = [apply_t(img) for img in s1_img_list]
#         s2_img_list = [apply_t(img) for img in s2_img_list]
#         mask_aug = apply_t(mask)

#         return s1_img_list, s2_img_list, mask_aug

def test_eu():
    s1_transform = transforms.Compose([NormalizeS1(),myToTensor()])
    s2_transform = transforms.Compose([NormalizeS2(),myToTensor()])
    crop_map_transform = transforms.Compose([CropMapTransform(),myToTensor(dtype=torch.int16)])
    
    print("Testing Dataset...")
    s1s2_dataset = Sen12Dataset(s1_dir="D:\\python\\CropMapping\\dataset\\ts_dataset_patched\\s1\\",
                                s2_dir="D:\\python\\CropMapping\\dataset\\ts_dataset_patched\\s2\\",
                                crop_map_dir="D:\\python\\CropMapping\\dataset\\ts_dataset_patched\\crop_map\\",
                                s1_transform=s1_transform,
                                s2_transform=s2_transform,
                                crop_map_transform=crop_map_transform,
                                verbose=True)

    print(f"Dataset length: {len(s1s2_dataset)}")
    print(f"s1_img type: {type(s1s2_dataset[0][0])}")
    print(f"s2_img type: {type(s1s2_dataset[0][1])}")
    print(f"crop_map type: {type(s1s2_dataset[0][2])}")
    print(f"s1_img shape: {s1s2_dataset[0][0].shape}")
    print(f"s2_img shape: {s1s2_dataset[0][1].shape}")
    print(f"crop_map shape: {s1s2_dataset[0][2].shape}")
    
def test_iran():
    s1_transform = transforms.Compose([NormalizeS1(),myToTensor()])
    s2_transform = transforms.Compose([NormalizeS2(),myToTensor()])
    crop_map_transform = transforms.Compose([CropMapTransformIran(),myToTensor(dtype=torch.int16)])
    augmentation = Augmentations()
    print("Testing Dataset...")
    s1s2_dataset = Sen12Dataset(s1_dir="D:\\Datasets\\crop_map_dataset_Iran_tomato\\s1\\",
                                s2_dir="D:\\Datasets\\crop_map_dataset_Iran_tomato\\\s2\\",
                                crop_map_dir="D:\\Datasets\\crop_map_dataset_Iran_tomato\\crop_map\\",
                                s1_transform=s1_transform,
                                s2_transform=s2_transform,
                                crop_map_transform=crop_map_transform,
                                augmentation=augmentation,
                                verbose=False)
    print("Creating Sampler Object...")
    sampler = BalancedSampler(s1s2_dataset, ratio=0.1, verbose=True)
    print("Creating DataLoader Object...")
    data_loader = torch.utils.data.DataLoader(s1s2_dataset, batch_size=1, sampler=sampler)
    print(f"Dataset length: {len(s1s2_dataset)}")
    print(f"Len of data_loader after sampling: {len(data_loader)}")
    print(f"s1_img type: {type(s1s2_dataset[0][0])}")
    print(f"s2_img type: {type(s1s2_dataset[0][1])}")
    print(f"crop_map type: {type(s1s2_dataset[0][2])}")
    print(f"s1_img shape: {s1s2_dataset[0][0].shape}")
    print(f"s2_img shape: {s1s2_dataset[0][1].shape}")
    print(f"crop_map shape: {s1s2_dataset[0][2].shape}")
    idx = 2
    s1s2 = s1s2_dataset[idx]
    img1 = s1s2[0][1,3,:,:]
    img2 = s1s2[1][1,3,:,:]
    img3 = s1s2[2][0,:,:]
    # plot in subplots
    plt.figure(figsize=(15,10))
    plt.subplot(131)
    plt.imshow(img1,cmap='gray')
    plt.subplot(132)
    plt.imshow(img2,cmap='gray')
    plt.subplot(133)
    plt.imshow(img3,cmap='gray')
    plt.show()
    
    

if __name__ == "__main__":
    from utils.plot_utils import *
    # print("TEST EU")
    # test_eu()
    print("TEST IRAN")
    test_iran()    
