# %%
import random, os, cv2

import numpy as np
import pandas as pd
import json
from datetime import datetime

from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import albumentations as A
import timm

from tqdm import tqdm
from PIL import Image

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau


import warnings
warnings.filterwarnings("ignore")

# %%
def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False

# %%
def get_scheduler(cfg, optimizer):
    scheduler = None
    if cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)
    elif cfg.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, 
                                         T_0 = CFG.epochs, 
                                         T_mult=1, 
                                         eta_min=1e-7, 
                                         last_epoch=-1, 
                                         verbose=False)
        
    elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=cfg.T_0,
                                                             eta_min=cfg.min_lr)
    return scheduler

# %%
# Config
class CFG:
    visible_folder  = "./dataset/processed/visibles/"
    infrared_folder = "./dataset/processed/infrareds/"
    mask_folder     = "./dataset/processed/masks/"
    label_file      = "./dataset/processed/label.csv"

    encoder_name   = 'resnet101' # resnet101, efficientnet-b6, timm-regnety_008, timm-regnety_120
    seg_model_name = 'metaseg' # FiLMedUnet, UNet
    activation     = None #softmax2d, sigmoid, softmax

    cutmix         = False # failed
    use_vi_inf     = True
    img_size       = 320
    scheduler      = None #"CosineAnnealingLR" #"ReduceLROnPlateau" #'CosineAnnealingWarmRestarts'
    epochs         = 12
    init_lr        = 0.0001
    min_lr         = 1e-6
    T_0            = 9
    T_mult         = 1
    batch_size     = 4
    weight_decay   = 1e-6
    
    depth          = 4
    film_layers    = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    seed           = 42
    n_fold         = 4
    train_kfold    = False
    train_fold     = [0, 1, 2, 3]

    num_class      = 4 # 4
    num_inputs     = 2 if use_vi_inf else 1
    use_meta       = True
    n_meta_features= 3

    save_folder    = f'results/{seg_model_name}_weights_{str(datetime.now().strftime("%m_%d_%Y-%H:%M:%S"))}/'
    save_weight_path     =  f'weights_{seg_model_name}_{num_inputs}_images_{use_meta}_meta.pth'

    device         = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

set_seed(CFG.seed)
if not os.path.exists(CFG.save_folder):
    os.makedirs(CFG.save_folder)

preprocessing_fn = lambda image : get_preprocessing_fn(encoder_name = CFG.encoder_name,
                                                       pretrained = 'imagenet')
preprocessing_fn = None
# %%

def Augment(mode):
    if mode == "train":
        train_aug_list = [ 
                        #   A.RandomScale(scale_limit=(1.2, 1.5), p=0.5), 
                          A.CenterCrop(CFG.img_size, CFG.img_size, p=1.0),
                          A.RandomRotate90(p=0.2), #0.75 fails
                          A.HorizontalFlip(p=0.5),
                          A.VerticalFlip(p=0.5),
                          
                        #   A.ChannelDropout(channel_drop_range=(1,2), p=0.2),

                          A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.2), rotate_limit=(-30,30), 
                         interpolation=1, border_mode=0, value=(0,0,0), p=0.2), #
                          A.OneOf([ #
                            A.GaussNoise(var_limit=(0,50.0), mean=0, p=0.5),
                            A.GaussianBlur(blur_limit=(3,7), p=0.5),
                            ], p=0.2),
                          A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.3, 
                                 brightness_by_max=True,p=0.5),
                          A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, 
                           val_shift_limit=0, p=0.5),

                        #   A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1), #
                        #   A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                        #   A.Cutout(max_h_size=20, max_w_size=20, num_holes=8, p=0.2),
                          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # default imagenet mean & std.
                          ]

        if CFG.use_vi_inf:
            return A.Compose(train_aug_list, #bbox_params=A.BboxParams(format="pascal_voc"),
                            additional_targets={'image2': 'image'}) # this is to augment both the normal and infrared sattellite images.
        else:
            return A.Compose(train_aug_list)#, bbox_params=A.BboxParams(format="pascal_voc"))
    else: # valid test
        valid_test_aug_list = [
                            # A.Resize(CFG.img_size, CFG.img_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
                            

        if CFG.use_vi_inf:
            return A.Compose(valid_test_aug_list,
                            additional_targets={'image2': 'image'})
        else:
            return A.Compose(valid_test_aug_list)

# %%
class FOREST(Dataset):
    def __init__(self,
                 label_df,
                 preprocess_input=None,
                 mode = "train" # train | valid | test
                ):
        self.label_df        = label_df
        self.mode            = mode
        self.preprocess_input = preprocess_input
        self.augment         = Augment(mode)
        self.augment2        = Augment('valid')
        self.mask_dict       = {"plantation"             : 1,
                                "grassland shrubland"    : 2,
                                "smallholder agriculture": 3,
                                "other"                  : 4}
          
    def __len__(self):        
        return len(self.label_df)
        
    def __getitem__(self, index):

        case_id, deforestation_type, lat, long, year, mode, data_path = self.label_df.iloc[index].to_list()
        
        # load image and mask
        visible  = cv2.imread(data_path + '/processed/visibles/'  + str(case_id) + "/composite.png")
        infrared = cv2.imread(data_path + '/processed/infrareds/' + str(case_id) + "/composite.png")
        mask     = cv2.imread(data_path + '/processed/masks/'     + str(case_id) + ".png", 0) if (self.mode != "test") else np.zeros(visible.shape[:2]) # dummy mask for test-set case.
        
        # convert the foreground region in the mask to the corressponding label integer
        label = self.mask_dict[deforestation_type]
        
        mask[mask == 1.] = label
        
        if CFG.use_vi_inf:
            # visible, infrared, mask, _, _, _, _
            visible, infrared, mask = self.augment(image  = visible,
                                                image2 = infrared, mask=mask).values()

            image = np.concatenate((visible, infrared), axis = -1)
        else:
            

            visible, mask = self.augment(image  = visible,
                                                mask   = mask).values()

            image = visible
        
        return torch.tensor(image), torch.tensor(mask), torch.tensor([year, lat, long]), label, case_id

# %%
def show_image(image,
               mask   = None,
               labels = ["no deforestation",
                         "plantation",
                         "grassland shrubland",
                         "smallholder agriculture",
                         "other"],
               colors = np.array([(0.,0.,0.),
                                  (0.667,0.,0.), 
                                  (0.,0.667,0.677), 
                                  (0.,0.,0.667),
                                  (0.667, 0.667, 0.667)])):
    
    # copy to prevent from modifying the input image and mask
    image = np.copy(image)
    mask  = np.copy(mask) if mask is not None else mask
    
    # normalize to [0-1]
    image = (image - image.min())/(image.max() - image.min())    
    # add good-looking color
    mask  = colors[mask] if mask is not None else mask
    
    plt.imshow(image, cmap='bone')    
    if mask is not None:
        plt.imshow(mask, alpha=0.6)
        handles = [Rectangle((0,0),1,1, color=color) for color in colors]
        plt.legend(handles, labels)
    plt.axis('off')
    
    return None

# %%
# Show Images
label_df = pd.read_csv(CFG.label_file)
label_df['data_folder'] = ['./dataset']*len(label_df)
train_df = label_df[label_df['mode'] == 'train']
val_df = label_df[label_df['mode'] == 'valid']
print(len(train_df), len(val_df))
# %%%
train_dataset = FOREST(train_df, mode = "train")

for i in range(20,35):
    image, mask, meta, _, case_id = train_dataset[i]
    visible = image[..., :3]
    
    print(case_id, torch.mean(mask.float()), meta)
    if CFG.use_vi_inf:
        show_image(visible, mask = mask)
    else:
        show_image(visible, mask = mask[0])
    plt.show()

# %%
# load models
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_channels = 3+3 if CFG.use_vi_inf else 3
print(f"Number of channels: {num_channels}")

from meta_models.film_model import FiLMedUnet, Unet
from meta_models.latefusion_model import Meta_Seg
if CFG.seg_model_name == "UNet":
    model = Unet(in_channel=num_channels, out_channel=CFG.num_class+1, depth=CFG.depth).to(CFG.device)
elif CFG.seg_model_name == 'FiLMedUnet':
    model = FiLMedUnet(in_channel=num_channels, out_channel=CFG.num_class+1, film_layers=CFG.film_layers, n_metadata=3, depth=CFG.depth).to(CFG.device)
elif CFG.seg_model_name == 'metaseg':
    seg_model = smp.UnetPlusPlus(
            encoder_name=CFG.encoder_name,      
            encoder_weights="imagenet",
            in_channels=num_channels,     
            classes=CFG.num_class+1,
            activation=CFG.activation)
    model = Meta_Seg(in_channels=num_channels, seg_model=seg_model, n_labels=CFG.num_class+1, n_meta_features=CFG.n_meta_features).to(CFG.device)
elif CFG.seg_model_name == "UNetPlusPlus":
    model = smp.UnetPlusPlus(
            encoder_name=CFG.encoder_name,      
            encoder_weights="imagenet",
            in_channels=num_channels,     
            classes=CFG.num_class+1,
            activation=CFG.activation).to(CFG.device)
    
print(count_parameters(model))


# %%
def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    true = true.unsqueeze(1)
    num_classes = logits.shape[1]
    device = 'cpu' if true.get_device() == -1 else f"cuda:{true.get_device()}"
    
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1).to(device)
        true_1_hot = true_1_hot[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes).to(device)
        true_1_hot = true_1_hot[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

# hard dice score for vadiation set evaluation
def hard_dice(pred, mask, label, eps=1e-7):

    #pick the channel that coressponds to the true label
    pred = (torch.argmax(pred, dim = 1) == label).long().view(-1)        
    mask = mask.view(-1)

    # compute hard dice score for the foreground region
    score = (torch.sum(pred * mask)*2)/ (torch.sum(pred) + torch.sum(mask) + eps)    
    
    return np.array(score)

alpha = 0.3 #0.3 #FP
beta = 1 - alpha # FN
gamma = 1.0
TverskyLoss = smp.losses.TverskyLoss(mode='multiclass', log_loss=False, alpha=alpha,\
                                      beta=beta, gamma=gamma)
DiceLoss    = smp.losses.DiceLoss(mode='multiclass')
LovaszLoss  = smp.losses.LovaszLoss(mode='multiclass', per_image=False)

# %%
loss_fn = TverskyLoss
CFG.init_lr = CFG.init_lr
# optimizer = optim.Adam(model.parameters(), lr=CFG.init_lr)
optimizer = optim.AdamW(model.parameters(), lr=CFG.init_lr)
# # learning rate scheduler
scheduler = get_scheduler(CFG, optimizer)

# %% cut mix rand bbox
def rand_bbox(size, lam, to_tensor=True):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    #uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    if to_tensor:
        bbx1 = torch.tensor(bbx1)
        bby1 = torch.tensor(bby1)
        bbx2 = torch.tensor(bbx2)
        bby2 = torch.tensor(bby2)

    return bbx1, bby1, bbx2, bby2
# %%
def train(trainloader, validloader, model, fold=0,
          n_epoch = 10):
    
    best_valid_dice = 0.
    for epoch in range(n_epoch):
        print("")
        model.train()
        train_loss = train_epoch(trainloader, model)        
        print(f"Epoch {epoch}/{n_epoch}, Train Loss: {train_loss}") #, LR: {scheduler.get_lr()}")
        
        with torch.no_grad():    
            valid_loss, valid_dice = evaluate_epoch(validloader, model)     
            print(f"Epoch {epoch}/{n_epoch}, Valid Loss: {valid_loss}, Valid Dice: {valid_dice}")
            # save model
            if best_valid_dice <= valid_dice:
                print("Saving...")
                best_valid_dice = valid_dice
                torch.save(model.state_dict(), f"./{CFG.save_folder}/{fold}_{valid_dice:.3f}_{CFG.save_weight_path}")
        
        match CFG.scheduler:
            case 'ReduceLROnPlateau':
                scheduler.step(valid_loss) # 
            case 'CosineAnnealingLR': #
                scheduler.step()
            case 'CosineAnnealingWarmRestarts': #
                scheduler.step()

    return model

# %%
def train_epoch(trainloader, model):
        
    losses = []
    
    for (inputs, targets, metas, *_) in tqdm(trainloader):
        # forward pass
        if CFG.cutmix and random.random() > 0.4:
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(inputs.size()[0])
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)    
            inputs[:, bbx1:bbx2, bby1:bby2, :] = inputs[rand_index, bbx1:bbx2, bby1:bby2, :]
            targets[:, bbx1:bbx2, bby1:bby2] = targets[rand_index, bbx1:bbx2, bby1:bby2]
        
        if CFG.seg_model_name == 'FiLMedUnet' or CFG.seg_model_name == 'metaseg':
            outputs = model.forward(inputs.permute(0,-1,1,2).to(CFG.device), metas.float().to(CFG.device))
        else:
            outputs = model.forward(inputs.permute(0,-1,1,2).to(CFG.device))
        
        targets = targets.long().to(CFG.device)
        # calculate loss
        loss = loss_fn(outputs, targets)

        # backward pass and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    
    return np.mean(losses)

 # %%
def evaluate_epoch(validloader, model):
    model.eval()
    scores = []
    loss = []
    for (inputs, targets, metas, label, _) in tqdm(validloader):    
        if CFG.seg_model_name == 'FiLMedUnet' or CFG.seg_model_name == 'metaseg':
            outputs = model.forward(inputs.permute(0,-1,1,2).to(CFG.device), metas.float().to(CFG.device))
        else:
            outputs = model.forward(inputs.permute(0,-1,1,2).to(CFG.device))
        outputs = outputs.detach().cpu()
        targets = targets.long()
        # calculate loss
        val_loss = loss_fn(outputs, targets)
        #calculate dice
        score = hard_dice(outputs, targets, label)

        loss.append(val_loss.item())
        scores.append(score)
    
    return np.mean(loss), np.mean(scores)

# %%
label_df = pd.read_csv(CFG.label_file)
label_df['data_folder'] = ['./dataset']*len(label_df)
print(f"Size of original df: {len(label_df)}")
print(label_df.head(5))
train_val_df = label_df
# %%
# generated_label_file = "./generated_dataset_2/processed/new_label.csv"
# generated_label_df   = pd.read_csv(generated_label_file, index_col=0)
# generated_label_df['data_folder'] = ['./generated_dataset_2']*len(generated_label_df)
# print(generated_label_df.head(5))
# print(f"Size of generated df: {len(generated_label_df)}")

# # combine them together
# label_df = pd.concat([label_df.reset_index(drop=True), generated_label_df.reset_index(drop=True)])#.reset_index(drop=True)
# print(f"Size of combined df: {len(label_df)}")
# label_df.head(5)

# %%
# Train Once
train_val_df = train_val_df[~train_val_df['mode'].isin(['test'])]
train_df = train_val_df[train_val_df['mode'] == 'train'][:100]
val_df = train_val_df[train_val_df['mode'] == 'valid'][:20]

train_dataset = FOREST(train_df, mode = "train")
valid_dataset = FOREST(val_df, mode = "valid")

# data loader
train_loader = DataLoader(train_dataset, 
                                  batch_size=CFG.batch_size, 
                                  num_workers=14,# sampler=sampler,
                                  shuffle=True, 
                                  pin_memory=True)
valid_loader = DataLoader(valid_dataset, 
                                batch_size=1, 
                                num_workers=8, 
                                shuffle=False,
                                pin_memory=False)


if not CFG.train_kfold:
    model = train(train_loader, valid_loader, model, fold=-1, n_epoch = CFG.epochs)


#%%
# Train k-Fold
if CFG.train_kfold:
    # Split your dataset into K-folds
    kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df)):
        if fold not in CFG.train_fold:
            continue
        
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        train_dataset = FOREST(train_df, mode='train')
                    
        train_loader = DataLoader(train_dataset, 
                                    batch_size=CFG.batch_size, 
                                    num_workers=14, 
                                    shuffle=True, 
                                    pin_memory=True)
        
        # # validation
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        val_dataset = FOREST(val_df, mode='valid')
            
        valid_loader = DataLoader(val_dataset, 
                                    batch_size=1, 
                                    num_workers=8, 
                                    shuffle=False,
                                    pin_memory=False)
        
        model = train(train_loader, valid_loader, model, fold=fold,
                n_epoch = CFG.epochs)
        
        print(f'Finish fold {fold}: Train size={len(train_df)}, Test size={len(val_df)}')

# %%