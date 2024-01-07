import ee
import ast
import geemap
import shutil
import importlib
import pandas as pd
from skimage import io
import utils.utils as utils
import utils.ee_utils as ee_utils
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import argparse

parser = argparse.ArgumentParser(description='Dataset Generator')

# Add arguments
parser.add_argument('--service-account', type=str, default='service',
                    help='Service account name, e.g. KEYNAME@PROJECTNAME.iam.gserviceaccount.com. Get GEE Google Cloud service account at: https://developers.google.com/earth-engine/guides/service_account')
parser.add_argument('--service-account-key-path', type=str, default='./service_acount_key.json', help='Path to service account key file (.json)')
parser.add_argument('--rois-path', type=str, default='../main_dataset/output/Iran_ROI.xlsx', help='Path to ROIs file')
parser.add_argument('--dataset-folder', type=str, default='./ts_dataset', help='Path to dataset folder')
parser.add_argument('--crop-type', type=str, default='onion', choices=['canola', 'cotton', 'lentils', 'maize', 'onion', 'pea', 'sugarbeet', 'tomato'],
                    help='Crop type to download')
parser.add_argument('--bands', nargs='+',
                    default=['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'],
                    help='List of bands')

args = parser.parse_args()

SERVICE_ACCOUNT = args.service_account
SERVICE_ACCOUNT_KEY_PATH = args.service_account_key_path
ROIS_PATH = args.rois_path
DATASET_FOLDER = args.dataset_folder
CROP_TYPE = args.crop_type
BANDS = args.bands

credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, SERVICE_ACCOUNT_KEY_PATH)


df = pd.read_excel(ROIS_PATH)
df = df[df['main_type'] == CROP_TYPE]
df = df.head(1).copy() # Only for testing.
print(f"Number of ROIs: {len(df)}")



mask_directory = f"{DATASET_FOLDER}/crop_map"
utils.check_folder_exists(mask_directory)

for i, row in df.iterrows():
    mask_source = f"../main_dataset/output/{row['main_type']}/masks/{row['uuid']}.png"
    mask_destination = f"{mask_directory}/{row['uuid']}.tif"
    if os.path.exists(mask_source):
        mask = io.imread(mask_source)
        io.imsave(mask_destination, mask)
    else:
        print(f"{mask_source} was not found. Search canceled.")
        break
else:
    print('All masks have been copied as TIF file to dataset directory.')
    

COUNTRY = 'Iran'


for i, row in df.iterrows():
    if row['country'] != COUNTRY:
        continue

    print(f"Processing {i + 1} of {len(df)} - {row['country']} - {row['main_type']}")

    r_roi = row['roi'].replace('\n', '').replace(' ', '')
    r_roi = ast.literal_eval(r_roi)
    r_roi = ee.Geometry.Polygon(r_roi, proj='EPSG:3857')
    r_date = row['harvest_date']
    r_name = f"{row['uuid']}.tif"

    ee_utils.ts_downloader(r_roi, r_name, r_date, scale=10, bands=BANDS, dataset_folder=DATASET_FOLDER)

utils.resize_masks(f'{DATASET_FOLDER}/s1/1/',f'{DATASET_FOLDER}/crop_map/', verbose=False)



main_in_folder =  DATASET_FOLDER
main_out_folder =  DATASET_FOLDER + '_patched'
utils.check_folder_exists(main_out_folder)


folders = ["/s1", "/s2"]
for folder in folders:
    ts_date_folders = os.listdir(main_in_folder + folder)
    for date in ts_date_folders:
        print("Processing: " + main_in_folder + folder + "//" + date)
        input_sat = "S2" if folder == "s2" else "S1"
        utils.patch_folder(main_in_folder + folder + "//" + date+ "//", main_out_folder + folder + "//" + date+ "//", input_sat=input_sat, remove_year=False, mute=True, no_overlap=False)

# patch crop map
print("Processing: " + main_in_folder + "/crop_map" + "//")
utils.patch_folder(main_in_folder + "/crop_map" + "//", main_out_folder + "/crop_map" + "//",
                   input_sat="crop_map", remove_year=False, mute=True, no_overlap=False)

print(f"Number of patches: {len(os.listdir(main_out_folder + '//s1//1'))}")

NAME = "crop_map_dataset_" + COUNTRY + "_" + CROP_TYPE