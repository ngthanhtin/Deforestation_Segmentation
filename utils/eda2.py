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
infrared_path = '../deep/downloads/ForestNetDataset/examples/-0.002226324002905_109.97159881327198/images/infrared/composite.npy'
visible_path = '../deep/downloads/ForestNetDataset/examples/-0.002226324002905_109.97159881327198/images/visible/composite.png'
mask_path = '../deep/downloads/ForestNetDataset/examples/-2.248346072674411_104.1357857482906/forest_loss_region.pkl'

# %%
np.load(infrared_path).shape

with open(mask_path, 'rb') as f:
    mask = pickle.load(f)

mask
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

# %%
combined_df = pd.concat([train_df, valid_df], axis=0)
combined_df.head(4)
len(combined_df)
# %%
combined_df.to_csv('add_label.csv', index=False)

# %%
import pandas as pd
df = pd.read_csv('/home/tin/projects/deforestation/Indonesia_Deforestation_Segmentation/dataset/processed/label.csv')
df.head(5)
# %%
value = df.loc[1]['id']
type(value)
# %% --check image
import os
import cv2
import numpy as np
import pandas as pd
path = '/home/tin/projects/deforestation/Indonesia_Deforestation_Segmentation/deep/downloads/ForestNetDataset/examples/'
folders = os.listdir(path)
df = pd.read_csv('/home/tin/projects/deforestation/Indonesia_Deforestation_Segmentation/dataset/add_label.csv')

ids = list(df['id'])
# %%
folders = [os.path.join(path, id) for id in ids]

for f in folders:
    visible = cv2.imread(os.path.join(f, 'images/visible/composite.png'))
    infrared = np.load(os.path.join(f, 'images/infrared/composite.npy'))    
    mask = cv2.imread(os.path.join(f, 'mask.png'), 0)
    print(mask.shape)
    print(type(visible))
# %%
