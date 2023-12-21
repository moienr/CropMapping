import torch 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_LIST = ["canola", "cotton", "lentils", "maize", "onion", "pea", "sugarbeet", "tomato"]