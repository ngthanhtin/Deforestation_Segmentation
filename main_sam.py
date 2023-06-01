# %%
import random, os, cv2

import numpy as np
import pandas as pd
import json

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

from PIL import Image

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler

from infrared_models.abcseg import Small_Segmentation_Model
from copy_paste_aug.copy_paste import CopyPaste

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
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler = None
    if cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
    elif cfg.scheduler == 'CosineAnnealingLR':
        scheduler_cosine = CosineAnnealingLR(optimizer, 
                                         T_0 = CFG.epochs, 
                                         T_mult=1, 
                                         eta_min=1e-7, 
                                         last_epoch=-1, 
                                         verbose=False)
        scheduler = GradualWarmupSchedulerV2(
            optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=cfg.T_0, 
                                                             eta_min=cfg.min_lr)
    return scheduler

# %%
# Config
class CFG:
    seg_model_name = 'SAM' # 
    activation     = None #softmax2d, sigmoid, softmax

    ensemble       = False
    use_vi_inf     = False
    img_size       = 320
    scheduler      = "CosineAnnealingWarmRestarts" #"CosineAnnealingLR" #"ReduceLROnPlateau" #'CosineAnnealingWarmRestarts'
    epochs         = 10
    init_lr        = 0.0005
    min_lr         = 1e-6
    T_0            = 25
    batch_size     = 4
    weight_decay   = 1e-6
    
    seed           = 42
    n_fold         = 4
    train_fold     = [0]

    num_class      = 4 # 4
    num_inputs     = 2 if use_vi_inf else 1

    save_folder = 'copy_paste_weights_2/'
    save_weight_path     =  f'weights_sam_{num_inputs}.pth'

    device         = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

set_seed(CFG.seed)
if not os.path.exists(CFG.save_folder):
    os.makedirs(CFG.save_folder)

preprocessing_fn = lambda image : get_preprocessing_fn(encoder_name = CFG.encoder_name,
                                                       pretrained = 'imagenet')
preprocessing_fn = None
# %%

def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

# %%
def Augment(mode):
    if mode == "train":
        train_aug_list = [ #A.RandomScale(scale_limit=(0.0, 1.0), p=0.5), 
                          A.CenterCrop(CFG.img_size, CFG.img_size, p=1.0),
                          A.RandomRotate90(p=0.2),
                          A.HorizontalFlip(p=0.5),
                          A.VerticalFlip(p=0.5),
                        
                          A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.2), rotate_limit=(-30,30), 
                         interpolation=1, border_mode=0, value=(0,0,0), p=0.2), #
                          A.OneOf([ #
                            A.GaussNoise(var_limit=(0,50.0), mean=0, p=0.5),
                            A.GaussianBlur(blur_limit=(3,7), p=0.5),
                            ], p=0.2),
                          A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, 
                                 brightness_by_max=True,p=0.5),
                          A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, 
                           val_shift_limit=0, p=0.5),

                        #   A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5), #
                        #   A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                        #   A.Cutout(max_h_size=20, max_w_size=20, num_holes=8, p=0.2),
                        # CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1),
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
                 processor,
                 mode = "train" # train | valid | test
                ):
        
        _label_df            = label_df  
        self.label_df        = _label_df
        self.mode            = mode
        self.augment         = Augment(mode)
        self.augment2        = Augment('valid')
        self.processor       = processor
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
        
        # mask[mask == 1.] = label
        
        # get bounding box prompt
        prompt = get_bounding_box(mask)
        mask = cv2.resize(mask, (256, 256))

        prompt = [prompt[0]*256/320, prompt[1]*256/320, prompt[2]*256/320, prompt[3]*256/320] # xmin, ymin, xmax, ymax
        
        if CFG.use_vi_inf:
            # visible, infrared, mask, _, _, _, _
            visible, infrared, mask = self.augment(image  = visible,
                                                image2 = infrared, mask=mask).values()
            
            visible_inputs = self.processor(visible, input_boxes=[[prompt]], return_tensors="pt")
            infrated_inputs = self.processor(infrared, input_boxes=[[prompt]], return_tensors="pt")
            # image = np.concatenate((visible, infrared), axis = -1)
            
            visible_inputs = {k:v.squeeze(0) for k,v in visible_inputs.items()}
            infrated_inputs = {k:v.squeeze(0) for k,v in infrated_inputs.items()}
            visible_inputs["ground_truth_mask"] = mask
            infrated_inputs["ground_truth_mask"] = mask

            return visible_inputs, infrated_inputs
        else:
            # visible, mask = self.augment(image  = visible,
            #                                     mask   = mask).values()
            visible_inputs = self.processor(visible, input_boxes=[[prompt]], return_tensors="pt")
            visible_inputs = {k:v.squeeze(0) for k,v in visible_inputs.items()}
            visible_inputs["ground_truth_mask"] = mask
        
            return visible_inputs

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
visible_folder  = "./dataset/processed/visibles/"
infrared_folder = "./dataset/processed/infrareds/"
mask_folder     = "./dataset/processed/masks/"
label_file      = "./dataset/processed/label.csv"

label_df = pd.read_csv(label_file)
label_df['data_folder'] = ['./dataset']*len(label_df)
train_df = label_df[label_df['mode'] == 'train']
val_df = label_df[label_df['mode'] == 'valid']

# %%%
from transformers import SamProcessor

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

train_dataset = FOREST(train_df, processor, mode = "train")

for i in range(0,10):
    inputs = train_dataset[i]
    image, mask = inputs["pixel_values"], inputs["ground_truth_mask"]
    # visible = image[..., :3]
    visible = torch.permute(image, (1,2,0))
    
    if CFG.num_inputs == 1:
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

from transformers import SamModel 
import monai

model = SamModel.from_pretrained("facebook/sam-vit-base").to(CFG.device)
# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

print(count_parameters(model))

# %%
# for module in model.modules():
#     # print(module)
#     if isinstance(module, nn.BatchNorm2d):
#         if hasattr(module, 'weight'):
#             module.weight.requires_grad_(False)
#         if hasattr(module, 'bias'):
#             module.bias.requires_grad_(False)
#         module.eval()
# load model
# model.load_state_dict(torch.load("./weights.pth"))

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
    pred = (torch.argmax(pred, dim = 1)).long().view(-1)        
    mask = mask.view(-1)

    # compute hard dice score for the foreground region
    score = (torch.sum(pred * mask)*2)/ (torch.sum(pred) + torch.sum(mask) + eps)    
    
    return np.array(score)

alpha = 0.3 #0.3 #FP
beta = 1 - alpha # FN
TverskyLoss = smp.losses.TverskyLoss(mode='multiclass', log_loss=False, alpha=alpha, beta=beta)
DiceLoss    = smp.losses.DiceLoss(mode='multiclass')
CELoss      = smp.losses.SoftCrossEntropyLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='multiclass', per_image=False)

seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
#%%
def focal_tversky(y_pred, y_true):
  pt_1 = TverskyLoss(y_pred, y_true)
  gamma = 0.3
  return torch.pow((1-pt_1), gamma)

# %%
loss_fn = seg_loss
CFG.init_lr = 0.0005
# optimizer = optim.Adam(model.parameters(), lr=CFG.init_lr)
optimizer = optim.AdamW(model.mask_decoder.parameters(), lr=CFG.init_lr)
# optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
# # learning rate scheduler
scheduler = get_scheduler(CFG, optimizer)

# optimizer = torch.optim.Adam(params=model.parameters(),
#                                  lr=1e-4,
#                                  weight_decay=1e-3)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
#                                                        gamma=0.95)

# %%
def train(trainloader, validloader, model, fold=0,
          n_epoch = 10):
    
    best_valid_dice = 0.
    for epoch in range(n_epoch):
        print("")
        model.train()
        train_loss = train_epoch(trainloader, model)        
        print(f"Epoch {epoch}/{n_epoch}, Train Loss: {train_loss}")
        
        with torch.no_grad():
            valid_loss, valid_dice = evaluate_epoch(validloader, model)     
            print(f"Epoch {epoch}/{n_epoch}, Valid Loss: {valid_loss}, Valid Dice: {valid_dice}")
            # save model
            if best_valid_dice <= valid_dice:
                print("Saving...")
                best_valid_dice = valid_dice
                torch.save(model.state_dict(), f"./{CFG.save_folder}/{fold}_{valid_dice:.3f}_{CFG.save_weight_path}")
        
    return model

def train_epoch(trainloader, model):
        
    losses = []
    
    for batch in trainloader:
        # forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(CFG.device),
                      input_boxes=batch["input_boxes"].to(CFG.device),
                      multimask_output=False)
        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(CFG.device)

        # calculate loss
        loss = loss_fn(predicted_masks, ground_truth_masks.unsqueeze(1))

        # backward pass and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        match CFG.scheduler:
            case 'ReduceLROnPlateau':
                scheduler.step(loss) # 
            case 'CosineAnnealingLR': #
                scheduler.step()
            case 'CosineAnnealingWarmRestarts': #
                scheduler.step()


        losses.append(loss.item())
    
    return np.mean(losses)

def evaluate_epoch(validloader, model):
    model.eval()
    scores = []
    loss = []
    for batch in validloader:
        outputs = model(pixel_values=batch["pixel_values"].to(CFG.device),
                      input_boxes=batch["input_boxes"].to(CFG.device),
                      multimask_output=False)
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(CFG.device)
        # calculate loss
        val_loss = loss_fn(predicted_masks, ground_truth_masks.unsqueeze(1))
        #calculate dice
        score = hard_dice(predicted_masks.detach().cpu(), ground_truth_masks.unsqueeze(1).detach().cpu(), 0)
        
        loss.append(val_loss.item())
        scores.append(score)
    
    return np.mean(loss), np.mean(scores)

# %%
visible_folder  = "./dataset/processed/visibles/"
infrared_folder = "./dataset/processed/infrareds/"
mask_folder     = "./dataset/processed/masks/"
label_file      = "./dataset/processed/label.csv"
label_df        = pd.read_csv(label_file)

label_df = pd.read_csv(label_file)
label_df['data_folder'] = ['./dataset']*len(label_df)
print(f"Size of original df: {len(label_df)}")
print(label_df.head(5))

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

# Split your dataset into K-folds
kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
# train_val_df = label_df[label_df["mode"].isin(['train', 'valid'])]
# train_val_df.to_csv("train_val_df_3.csv")
# train_val_df = pd.read_csv("train_val_df_2.csv", index_col=0)
# train_val_df.tail(5)
train_val_df = label_df

# %%
# Train Once
print(len(train_val_df))
train_df = train_val_df[train_val_df['mode'] == 'train']
val_df = train_val_df[train_val_df['mode'] == 'valid']

train_dataset = FOREST(train_df, processor, mode = "train")
valid_dataset = FOREST(val_df, processor, mode = "valid")

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



model = train(train_loader, valid_loader, model, fold=4,
              n_epoch = 12)


#%%
# Train k-Fold
for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df)):
    # if fold != CFG.train_fold:
    #     continue
    
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
# Test on real validation set
visible_folder  = "./dataset/processed/visibles/"
infrared_folder = "./dataset/processed/infrareds/"
mask_folder     = "./dataset/processed/masks/"
label_file      = "./dataset/processed/label.csv"
label_df        = pd.read_csv(label_file)
label_df['data_folder'] = ['./dataset']*len(label_df)

train_df = label_df[label_df['mode'] == 'train']
val_df = label_df[label_df['mode'] == 'valid']

train_dataset = FOREST(train_df, mode = "train")
valid_dataset = FOREST(val_df, mode = "valid")

print(f"Len of full train dataset: {len(train_dataset)}")
print(f"Len of full valid dataset: {len(valid_dataset)}")
# train_loader = DataLoader(train_dataset,
#                           batch_size  = CFG.batch_size,
#                           num_workers = 14,
#                           shuffle     = True, 
#                           pin_memory  = True)

valid_loader = DataLoader(valid_dataset,
                          batch_size  = 1,
                          num_workers = 8,
                          shuffle     = False,
                          pin_memory  = False)

weight_dir = 'copy_paste_weights_3/'
weight_paths = os.listdir(weight_dir)
# weight_paths = ['3_0.332_weights_dice_resnet101_UNetPlusPlus_1images.pth']
weight_paths.sort(key=lambda x: x[0])
weight_paths = [os.path.join(weight_dir, p) for p in weight_paths]

for path in weight_paths:
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.no_grad():    
        print(f"Inference model: {path}")
        valid_loss, valid_dice = evaluate_epoch(valid_loader, model)
        print(f"Valid Loss: {valid_loss}, Valid Dice: {valid_dice}, LB Score: {valid_dice-0.086}")


# %%
#----------------SUBMISSION-------------------#
# lets define mask to RLE conversion
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    
    # to string format
    runs = ' '.join(str(x) for x in runs)
    
    return runs

def predict(model, loader):
    
    test_results = []
    for (inputs, _, label, image_id) in loader:
        
        # forward pass       
        pred = model(inputs.permute(0,-1,1,2).to(CFG.device)) # channel first

        # move back to cpu
        pred     = pred.detach().cpu()
        image_id = str(image_id[0].item())
        
        #pick the channel that coressponds to the true label
        pred = (torch.argmax(pred, dim = 1) == label).squeeze(0).long().numpy()
                
        #convert to rle
        pred_rle = rle_encode(pred)
        
        test_results.append({"image_id" : image_id,
                             "pred_rle" : pred_rle})
        
    return test_results

# class EnsembleModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.ModuleList()
#         for fold in [1, 2, 3]:
#             _model = build_model(CFG, weight=None)
#             #_model.to(device)

#             model_path = f'/{CFG.train_fold}/Unet_fold{fold}_best.pth'
#             state = torch.load(model_path)['model']
#             _model.load_state_dict(state)
#             _model.eval()

#             self.model.append(_model)
    
#     def forward(self, x):
#         output=[]
#         for m in self.model:
#             output.append(m(x))
#         output=torch.stack(output,dim=0).mean(0)
#         return output

# import tensor_comprehensions as tc
# def TTA(x:tc.Tensor,model:nn.Module):
#     #x.shape=(batch,c,h,w)
#     if CFG.TTA:
#         shape=x.shape
#         x=[x,*[tc.rot90(x,k=i,dims=(-2,-1)) for i in range(1,4)]]
#         x=tc.cat(x,dim=0)
#         x=model(x)
#         x=torch.sigmoid(x)
#         x=x.reshape(4,shape[0],*shape[2:])
#         x=[tc.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]
#         x=tc.stack(x,dim=0)
#         return x.mean(0)
#     else :
#         x=model(x)
#         x=torch.sigmoid(x)
#         return x
    
# %%
visible_folder  = "./dataset/processed/visibles/"
infrared_folder = "./dataset/processed/infrareds/"
mask_folder     = "./dataset/processed/masks/"
label_file      = "./dataset/processed/label.csv"
label_df        = pd.read_csv(label_file)

test_df = label_df[label_df["mode"].isin(['test'])]
test_df['data_folder'] = ['./dataset']*len(test_df)
test_df.head(10)

# %%
test_dataset = FOREST(test_df,
                      mode = "test")

test_loader = DataLoader(test_dataset,
                         batch_size  = 1,
                         num_workers = 14,
                         shuffle     = False,
                         pin_memory  = False)

# %%
# load model
model.load_state_dict(torch.load("./copy_paste_weights_2/4_0.330_weights_dice_resnet101_UNetPlusPlus_2images.pth"))

test_results = predict(model, test_loader)

df_submission = pd.DataFrame.from_dict(test_results)

df_submission.to_csv("my_submission.csv", index = False)
# %%
