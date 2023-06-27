# %%
import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A

# %%
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img

def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
# %%
image = cv2.imread("/home/tin/deforestation/dataset/processed/visibles/9684487/composite.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

bboxes = [[5.66, 138.95, 147.09, 164.88]]#, [366.7, 80.84, 132.8, 181.84]]
category_ids = [17]#, 18]

# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image
category_id_to_name = {17: 'smallholder', 18: 'fire'}

# %%
visualize(image, bboxes, category_ids, category_id_to_name)

# %%
transform = A.Compose(
    [A.CenterCrop(height=280, width=280, p=1)],
    bbox_params=A.BboxParams(format='coco', min_visibility=0.3, label_fields=['category_ids']),
)
random.seed(7)
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,
)
# %%
# --- test copy paste----

def visualize_2(image, bboxes):
    img = image.copy()
    for bbox in bboxes:
        class_name = bbox[-1]
        img = visualize_bbox(img, bbox[:-1], class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)

# %%
from copy_paste_aug.copy_paste import CopyPaste
import numpy as np

transform = A.Compose(
    [#A.Resize(600, 600), 
    #  A.CenterCrop(height=576, width=576, p=1), 
     CopyPaste(blend=True, sigma=1, pct_objects_paste=1, p=1, max_paste_objects=1)],
    bbox_params=A.BboxParams(format='coco')#, min_visibility=0.3, label_fields=['category_ids']),
)

random.seed(7)

image = cv2.imread("/home/tin/deforestation/dataset/processed/visibles/3847468/composite.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread("/home/tin/deforestation/dataset/processed/masks/3847468.png", 0)
h,w,_ = image.shape
mask = np.zeros((h,w))
mask[138:138+165, 5+147] = 1.

bboxes = [[5.66, 138.95, 147.09, 164.88, "smallholder" ]]#, [366.7, 80.84, 132.8, 181.84]]

# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image


random_image = cv2.imread("/home/tin/deforestation/dataset/processed/visibles/9684487/composite.png")
random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
random_mask = cv2.imread("/home/tin/deforestation/dataset/processed/masks/9684487.png", 0)
h,w,_ = random_image.shape
random_mask = np.zeros((h,w))
random_mask[100:100+114, 3+127] = 1.

random_bboxes = [[150.66, 50.95, 120.09, 120.88, "random"]]#, [366.7, 80.84, 132.8, 181.84]]

transformed = transform(image=image,
                        bboxes=bboxes,
                        masks = [mask],
                        paste_image=random_image,
                        paste_masks=[random_mask],                 
                        paste_bboxes=random_bboxes,)


print(np.array_equal(mask, transformed['masks'][0]))
print(np.array_equal(image, transformed['image']))

visualize_2(
    image,
    bboxes,
)

visualize_2(
    transformed['image'],
    transformed['bboxes'],
)

visualize_2(
    random_image,
    random_bboxes,
)
# %%
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

import numpy as np
import torch


def mixup_data(x, y, alpha=1.0, device='cpu'):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
        # lam = min(lam, 1-lam)
    else:
        lam = 1.
    batch_size = x.size()[0]
 
    index = torch.randperm(batch_size).to(device)


    ## SYM
    # mixed_x = lam * x + (1 - lam) * x[index,:]
    # mixed_y = (1 - lam) * x + lam * x[index,:]
    # mixed_image  = torch.cat([mixed_x,mixed_y], 0)
    # y_a, y_b = y, y[index]
    # mixed_label  = torch.cat([y_a,y_b], 0)


    ## Reduce batch size
    # new_batch_size = batch_size // 2
    # x_i = x[ : new_batch_size]
    # x_j = x[new_batch_size : ]
    # y_a = y[ : new_batch_size]
    # y_b = y[new_batch_size : ]
    # mixed_x = lam * x_i + (1 - lam) * x_j

    ## NO SYM
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]

    ## Only Alpha
    # mixed_x = 0.5 * x + (1 - 0.5) * x[index,:]
    # mixed_image  = mixed_x
    # y_a, y_b = y, y[index]
    # ind_label = torch.randint_like(y, 0,2)
    # mixed_label  = ind_label * y_a + (1-ind_label) * y_b

    ## Reduce batch size and SYM
    # new_batch_size = batch_size // 2
    # x_i = x[ : new_batch_size]
    # x_j = x[new_batch_size : ]
    # y_a = y[ : new_batch_size]
    # y_b = y[new_batch_size : ]
    # mixed_x = lam * x_i + (1 - lam) * x_j
    # mixed_y = (1 - lam) * x_i + lam * x_j
    # mixed_x  = torch.cat([mixed_x,mixed_y], 0)
    # y_b = torch.cat([y_b,y_a], 0)
    # y_a = y


    # return mixed_image, mixed_label, lam
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    # sigmoid = 1.0/(1 + math.exp( 5 - 10*lam))
    # sigmoid = 4.67840515/(5.85074311 + math.exp(6.9-10.2120858*lam))
    # sigmoid = 1.531 /(1.71822 + math.exp(6.9-12.2836*lam))
    # return lambda criterion, pred: sigmoid * criterion(pred, y_a) + (1 - sigmoid) * criterion(pred, y_b)

    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# %%
import cv2
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

def show_image(image):
    """
    image: (H,W, 3)
    """
    # normalize to [0-1]
    image = (image - image.min())/(image.max() - image.min())        
    plt.imshow(image, cmap='bone')    
        
    plt.axis('off')

image1 = cv2.imread("/home/tin/deforestation/dataset/processed/visibles/9684487/composite.png")
image2 = cv2.imread("/home/tin/deforestation/dataset/processed/visibles/3418608/composite.png")
image3 = cv2.imread("/home/tin/deforestation/dataset/processed/visibles/3847468/composite.png")
image4 = cv2.imread("/home/tin/deforestation/dataset/processed/visibles/3915278/composite.png")
image5 = cv2.imread("/home/tin/deforestation/dataset/processed/visibles/14353252/composite.png")


transform = transforms.ToTensor()

tensor1 = transform(image1)
tensor2 = transform(image2)
tensor3 = transform(image3)
tensor4 = transform(image4)
tensor5 = transform(image5) 

# %%
x = torch.stack([tensor1, tensor2, tensor3, tensor4, tensor5], dim=0)
y = torch.zeros(5)
mixed_x, y_a, y_b, lam = mixup_data(x, y, 0.5, device='cpu')
y_a, y_b, mixed_x.shape


# %%
for i in range(5):
    image = mixed_x[i]
    image = torch.permute(image, (2,1,0))
    image = np.copy(image)
    show_image(image)
    plt.show()
# %%
