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
df = pd.read_csv('../dataset/processed/label.csv')
df
# %%
label_list = set(df.merged_label.values.tolist())
label_list

# %%
train_df = df[df["mode"] == 'train']
valid_df = df[df["mode"] == 'valid']
test_df = df[df["mode"] == 'test']
#case_id, deforestation_type, lat, long, year, _ =train_df.iloc[0].to_list()
train_df
# %%
MASK_DICT = {"plantation"             : 1,
            "grassland shrubland"    : 2,
            "smallholder agriculture": 3,
            "other"                  : 4}


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
train_image_names = list(train_df['id'])
valid_image_names = list(valid_df['id'])
test_image_names = list(test_df['id'])

len(train_image_names), len(valid_image_names), len(test_image_names)
 #%%
# Check if any image does not contain masks
mask_folder = '../dataset/processed/masks/'
path = os.listdir(mask_folder)

# %%
train_path = [os.path.join(mask_folder, p) for p in path if int(p[:-4]) in train_image_names]
valid_path = [os.path.join(mask_folder, p) for p in path if int(p[:-4]) in valid_image_names]
test_path = [os.path.join(mask_folder, p) for p in path if int(p[:-4]) in test_image_names]
len(train_path), len(valid_path), len(test_path)

# %%


eliminate_train_id = []

for p in train_path:
    mask = cv2.imread(p, 0)
    if np.mean(mask) <= 1e-4:
        # print(np.count_nonzero(mask == 1))
        eliminate_train_id.append(int(p.split('/')[-1][:-4]))

    flatten_mask = np.ravel(mask)
    all_0 = np.any(flatten_mask == 1)
len(eliminate_train_id)

# %%
eliminate_valid_id = []

for p in valid_path:
    mask = cv2.imread(p, 0)
    if np.mean(mask) <= 1e-4:
        # print(np.count_nonzero(mask == 1))
        eliminate_valid_id.append(int(p.split('/')[-1][:-4]))
        
    flatten_mask = np.ravel(mask)
    all_0 = np.any(flatten_mask == 1)
len(eliminate_valid_id)
# %% remove those small-pixel samples

df = pd.read_csv('../dataset/processed/label.csv')
len(df), eliminate_train_id[:4], eliminate_valid_id[:4], df['id'][:4]
# %%
eliminate_id_list = eliminate_train_id + eliminate_valid_id
filtered_df = df[~df['id'].isin(eliminate_id_list)] 
len(filtered_df)
# %%
filtered_df.to_csv('label_remove_small_pixels.csv', index=False)
# %%
