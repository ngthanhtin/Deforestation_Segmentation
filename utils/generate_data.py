# %%
import pandas as pd
import cv2, os
import numpy as np
from Indonesia_Deforestation_Segmentation.utils.copy_paste_segmentation import copy_paste
from PIL import Image

# GENERATED FOLDER
GENERATED_FOLDER = 'generated_dataset_3/'
NUM_SAMPLES = 200
if not os.path.exists(GENERATED_FOLDER):
    os.makedirs(GENERATED_FOLDER)
    os.makedirs(f'{GENERATED_FOLDER}/processed/visibles/')
    os.makedirs(f'{GENERATED_FOLDER}/processed/infrareds/')
    os.makedirs(f'{GENERATED_FOLDER}/processed/masks/')


#%%
orig_data_path = './dataset/processed/'
visible_folder = os.path.join(orig_data_path, 'visibles')
infrared_folder = os.path.join(orig_data_path, 'infrareds')
mask_folder = os.path.join(orig_data_path, 'masks')
label_path = os.path.join(orig_data_path, 'label.csv')

label_df = pd.read_csv(label_path)
train_val_df = label_df[label_df["mode"].isin(['train', 'valid'])]
train_val_df = train_val_df.sample(n=NUM_SAMPLES)
train_val_df = train_val_df.reset_index(drop=True)
print(len(train_val_df[train_val_df['mode'] == 'train']))
print(len(train_val_df[train_val_df['mode'] == 'valid']))
data_size = len(train_val_df)
train_val_df.head(10)

#%%
data_size
#%%
new_label_dict = {'id':[], 'merged_label':[], 'latitude':[], 'longitude':[], 'year':[], 'mode':[]}

for i in range(data_size):
    # src sample
    src_case_id, src_deforestation_type, _, _, _, src_mode = train_val_df.iloc[i].to_list()
    if src_mode == 'valid':
        continue
    if src_deforestation_type not in ['grassland shrubland', "other"]:
        continue
    src_visible  = cv2.imread(visible_folder + f"/{str(src_case_id)}/composite.png")
    src_infrared = cv2.imread(infrared_folder + f"/{str(src_case_id)}/composite.png")
    src_mask     = cv2.imread(mask_folder + f"/{str(src_case_id)}.png", 0)

    # main sample
    for j in range(data_size):
        if i == j:
            continue
        
       
        main_case_id, main_deforestation_type, main_lat, main_long, main_year, main_mode = train_val_df.iloc[j].to_list()
        if src_deforestation_type != main_deforestation_type:
            continue
        if main_mode == 'valid':
            continue
        main_visible  = cv2.imread(visible_folder + f"/{str(main_case_id)}/composite.png")
        main_infrared = cv2.imread(infrared_folder + f"/{str(main_case_id)}/composite.png")
        main_mask     = cv2.imread(mask_folder + f"/{str(main_case_id)}.png", 0)

        # Copy-Paste data augmentation
        mask, generated_visible = copy_paste(src_mask, src_visible, main_mask, main_visible, lsj=True)
        mask, generated_infrared = copy_paste(src_mask, src_infrared, main_mask, src_infrared, lsj=True)

        #
        new_case_id = f'{main_case_id}_{src_case_id}'
        label = main_deforestation_type
        lat, long, year, mode = main_lat, main_long, main_year, 'train'

        # save_colored_mask(mask, mask_filename)
        os.makedirs(f'{GENERATED_FOLDER}/processed/visibles/{new_case_id}')
        os.makedirs(f'{GENERATED_FOLDER}/processed/infrareds/{new_case_id}')

        cv2.imwrite(f'{GENERATED_FOLDER}/processed/visibles/{new_case_id}/composite.png', generated_visible)
        cv2.imwrite(f'{GENERATED_FOLDER}/processed/infrareds/{new_case_id}/composite.png', generated_infrared)
        cv2.imwrite(f'{GENERATED_FOLDER}/processed/masks/{new_case_id}.png', mask)

        new_label_dict['id'].append(new_case_id) 
        new_label_dict['merged_label'].append(label) 
        new_label_dict['latitude'].append(lat) 
        new_label_dict['longitude'].append(long) 
        new_label_dict['year'].append(year)  
        new_label_dict['mode'].append(mode)

new_label_df = pd.DataFrame.from_dict(new_label_dict)
new_label_df.to_csv(f'{GENERATED_FOLDER}/processed/new_label.csv') 

# %%
#-----GENERATE BOX----
import numpy as np
import cv2
import pandas as pd
import json

def extract_box_from_mask(seg_path):
    im = cv2.imread(str(seg_path), 0)

    seg_value = 1

    np_seg = np.array(im)
    segmentation = np.where(np_seg == seg_value)

    # Bounding Box
    bbox = 0, 0, 0, 0
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

        bbox = x_min, x_max, y_min, y_max
    else:
        print("Error: Segmentation image could not be read or is empty.")
    
    return bbox

# %%
seg_path = './dataset/processed/masks/'
label_path = os.path.join('./dataset/processed/', 'label.csv')

label_df = pd.read_csv(label_path)
train_val_df = label_df[label_df["mode"].isin(['train', 'valid'])]
train_val_df = train_val_df.reset_index(drop=True)

data_size = len(train_val_df)

box_dict = {}
for i in range(data_size):
    case_id, _, _, _, _, _ = train_val_df.iloc[i].to_list()
    mask_path = os.path.join(seg_path, f'{str(case_id)}.png')

    x_min, x_max, y_min, y_max = extract_box_from_mask(mask_path)
    box_dict[str(case_id)] = [x_min, x_max, y_min, y_max]

with open("boxes.json", "w") as outfile:
    json.dump(box_dict, outfile, indent=4)
# %%
