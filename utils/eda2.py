# %%
import random, os, cv2

import numpy as np
import pandas as pd

from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


import warnings
warnings.filterwarnings("ignore")

# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import os

# %%
train_df = pd.read_csv('../deep/downloads/ForestNetDataset/train.csv')
valid_df = pd.read_csv('../deep/downloads/ForestNetDataset/val.csv')
test_df = pd.read_csv('../deep/downloads/ForestNetDataset/test.csv')
len(train_df), len(valid_df), len(test_df)
# %%
label_list = set(train_df.merged_label.values.tolist())
label_list

#case_id, deforestation_type, lat, long, year, _ =train_df.iloc[0].to_list()
# %%
MASK_DICT = {"Plantation"             : 1,
            "Grassland shrubland"    : 2,
            "Smallholder agriculture": 3,
            "Other"                  : 4}


# %%
for x in MASK_DICT.keys():
    sub_df = train_df.loc[train_df.merged_label==x, :]
    print(f"In Train Set: {x} has {len(sub_df)} samples")
# %%
for x in MASK_DICT.keys():
    sub_df = valid_df.loc[valid_df.merged_label==x, :]
    print(f"In Valid Set: {x} has {len(sub_df)} samples")
# %%
for x in MASK_DICT.keys():
    sub_df = test_df.loc[test_df.merged_label==x, :]
    print(f"In Test Set: {x} has {len(sub_df)} samples")
# %%
train_image_names = list(train_df['example_path'])
valid_image_names = list(valid_df['example_path'])
test_image_names = list(test_df['example_path'])

len(train_image_names), len(valid_image_names), len(test_image_names)
 #%%
# Check if any image does not contain masks
root_folder = '../deep/downloads/ForestNetDataset/'

# %%
train_path = [os.path.join(root_folder, p) for p in train_image_names]
valid_path = [os.path.join(root_folder, p) for p in valid_image_names]
test_path = [os.path.join(root_folder, p) for p in test_image_names]
len(train_path), len(valid_path), len(test_path), train_path[:2]

# %%
import pickle
import shapely
from shapely.geometry import shape
import fiona
from PIL import Image, ImageDraw

def read_mask(mask_path):
    with open(mask_path, 'rb') as f:
        mask = pickle.load(f)
    
    mask_image = Image.new('L', (332, 332), color=0)
    mask_geo = None
    if isinstance(mask, shapely.geometry.polygon.Polygon):
        print("The variable is a Polygon.")
        mask_geo = 'polygon'
        # Create a drawing context
        draw = ImageDraw.Draw(mask_image)

        # Draw polygons on the mask image
        poly = mask
        exterior_coords = list(poly.exterior.coords)
        interior_coords = [list(interior.coords) for interior in poly.interiors]
            
        draw.polygon(exterior_coords, fill=1)
        for interior in interior_coords:
            draw.polygon(interior, fill=0)

        # Convert the mask image to a NumPy array
        mask_array = np.array(mask_image)


    if isinstance(mask, shapely.geometry.multipolygon.MultiPolygon):
        print("The variable is a MultiPolygon.")
        mask_geo = 'multipolygon'
        # Create a drawing context
        draw = ImageDraw.Draw(mask_image)

        # Draw polygons on the mask image
        for poly in mask:
            exterior_coords = list(poly.exterior.coords)
            interior_coords = [list(interior.coords) for interior in poly.interiors]
            
            draw.polygon(exterior_coords, fill=1)
            for interior in interior_coords:
                draw.polygon(interior, fill=0)

        # Convert the mask image to a NumPy array
        mask_array = np.array(mask_image)

    return mask_array, mask_geo
    


# %%
import cv2
for train_image_path in train_path:
    pkl_mask = train_image_path + '/forest_loss_region.pkl'
    np_mask, geo_type = read_mask(pkl_mask)
    cv2.imwrite(os.path.join(train_image_path, 'mask.png'), np_mask)
    # Display the mask image
    # if geo_type == 'polygon':
    #     plt.imshow(np_mask, cmap='gray')
    #     plt.axis('off')
    #     plt.show()
    #     print(pkl_mask)
    #     break
# %%
for val_image_path in valid_path:
    pkl_mask = val_image_path + '/forest_loss_region.pkl'
    np_mask, geo_type = read_mask(pkl_mask)
    cv2.imwrite(os.path.join(val_image_path, 'mask.png'), np_mask)

# %%
for test_image_path in test_path:
    pkl_mask = test_image_path + '/forest_loss_region.pkl'
    np_mask, geo_type = read_mask(pkl_mask)
    cv2.imwrite(os.path.join(test_image_path, 'mask.png'), np_mask)
# %%
train_df = train_df.drop('label', axis=1)
train_df.head(4)
# %%
train_df['example_path'] = train_df['example_path'].str.replace('examples/', '')
train_df.head(5) 
# %%
train_df['merged_label'] = train_df['merged_label'].str.lower()
train_df.head(5) 
# %%
train_df = train_df.rename(columns={'example_path': 'id'})
train_df.head(5) 
# %% --valid df--
valid_df = valid_df.drop('label', axis=1)
valid_df.head(4)
# %%
valid_df['example_path'] = valid_df['example_path'].str.replace('examples/', '')
valid_df.head(5) 
# %%
valid_df['merged_label'] = valid_df['merged_label'].str.lower()
valid_df.head(5) 
# %%
valid_df = valid_df.rename(columns={'example_path': 'id'})
valid_df.head(5) 
# %% test df
test_df = test_df.drop('label', axis=1)
test_df.head(4)
# %%
test_df['example_path'] = test_df['example_path'].str.replace('examples/', '')
test_df.head(5) 
# %%
test_df['merged_label'] = test_df['merged_label'].str.lower()
test_df.head(5) 
# %%
test_df = test_df.rename(columns={'example_path': 'id'})
test_df.head(5) 
# %%

# %%
combined_df = pd.concat([train_df, valid_df, test_df], axis=0)
combined_df.head(4)
len(combined_df)
# %%
combined_df.to_csv('add_label_v2.csv', index=False)
# %%
