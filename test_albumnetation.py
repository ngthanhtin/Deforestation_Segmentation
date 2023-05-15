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
    [A.CenterCrop(height=280, width=280, p=1), 
     CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1)],
    bbox_params=A.BboxParams(format='coco')#, min_visibility=0.3, label_fields=['category_ids']),
)

random.seed(7)

image = cv2.imread("/home/tin/deforestation/dataset/processed/visibles/9684487/composite.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread("/home/tin/deforestation/dataset/processed/masks/9684487.png", 0)
h,w,_ = image.shape
mask = np.zeros((h,w))
mask[138:138+165, 5+147] = 1.

bboxes = [[5.66, 138.95, 147.09, 164.88, "smallholder" ]]#, [366.7, 80.84, 132.8, 181.84]]

# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image


random_image = cv2.imread("/home/tin/deforestation/dataset/processed/visibles/3847468/composite.png")
random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
random_mask = cv2.imread("/home/tin/deforestation/dataset/processed/masks/3847468.png", 0)
h,w,_ = random_image.shape
random_mask = np.zeros((h,w))
random_mask[100:100+114, 3+127] = 2.

random_bboxes = [[150.66, 50.95, 120.09, 120.88, "random"]]#, [366.7, 80.84, 132.8, 181.84]]

transformed = transform(image=image,
                        bboxes=bboxes,
                        masks = [mask],
                        paste_image=random_image,
                        paste_masks=[random_mask],                 
                        paste_bboxes=random_bboxes,)

print(type(transformed['paste_image']))


visualize_2(
    transformed['image'],
    transformed['bboxes'],
)

# %%
