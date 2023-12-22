import torch 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_LIST = ["canola", "cotton", "lentils", "maize", "onion", "pea", "sugarbeet", "tomato"]

# Dataset Dirs must be in the same order as CROP_LIST
# If you don't have all the datasets, make their path an empty string.
DATASET_DIRS = ["./irandatasetcanolav1/",
                "./irandatasetcottonv1/",
                "./irandatasetlentilsv1/",
                "./irandatasetmaizev1/",
                "./irandatasetonionv1/",
                "./irandatastpeav1/",
                "./irandatasetsugarbeetv1/",
                "./irandatasettomatov1/"]
# DATASET_DIRS = ["",
#                 "",
#                 "",
#                 "",
#                 "",
#                 "",
#                 "",
#                 "D://Datasets/crop_map_dataset_Iran_tomato-small"]

# This makes sure that all the dataset paths end with a slash
DATASET_DIRS = [dir_path + "/" if dir_path and dir_path[-1] != "/" else dir_path for dir_path in DATASET_DIRS]