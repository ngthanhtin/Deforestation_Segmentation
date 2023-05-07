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
df = pd.read_csv('./dataset/processed/label.csv')
df
# %%
train_df = df[df["mode"] == 'train']
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
    sub_df = test_df.loc[test_df.merged_label==x, :]
    print(f"In Test Set: {x} has {len(sub_df)} samples")
# %%
