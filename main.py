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
from sklearn.model_selection import KFold

import segmentation_models_pytorch as smp
import albumentations as A
import timm

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler

from infrared_models.uiunet import UIUNET
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
    elif cfg.scheduler == 'CosineAnnealingLR':
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.epochs, eta_min=1e-7)
        scheduler = GradualWarmupSchedulerV2(
            optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=cfg.T_0, 
                                                             eta_min=cfg.min_lr)
    return scheduler

# %%
# Config
class CFG:
    encoder_name   = 'resnet101' # resnet101, efficientnet-b6, timm-regnety_008
    seg_model_name = 'UNetPlusPlus' # UNetPlusPlus, UIUNet, UNet

    ensemble       = False
    use_vi_inf     = False
    img_size       = 320
    scheduler      = "CosineAnnealingWarmRestarts" #"CosineAnnealingLR" #"ReduceLROnPlateau" #'CosineAnnealingWarmRestarts'
    epochs         = 10
    init_lr        = 0.0005
    min_lr         = 1e-6
    T_0            = 25
    batch_size     = 16
    weight_decay   = 1e-6
    
    seed           = 42
    n_fold         = 4
    train_fold     = [0]

    num_class      = 4
    save_weight_path     =  f'weights_dice_{encoder_name}.pth'

    device         = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

set_seed(CFG.seed)
# %%
def Augment(mode):
    if mode == "train":
        train_aug_list = [ #A.RandomScale(scale_limit=(0.0, 1.0), p=0.5), 
                          A.Resize(CFG.img_size, CFG.img_size),
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
                          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # default imagenet mean & std.
                          ]
        if CFG.use_vi_inf:
            return A.Compose(train_aug_list,
                            additional_targets={'image2': 'image'}) # this is to augment both the normal and infrared sattellite images.
        else:
            return A.Compose(train_aug_list)
    else: # valid test
        valid_test_aug_list = [A.Resize(CFG.img_size, CFG.img_size), 
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        if CFG.use_vi_inf:
            return A.Compose(valid_test_aug_list,
                            additional_targets={'image2': 'image'})
        else:
            return A.Compose(valid_test_aug_list)

class FOREST(Dataset):
    def __init__(self,
                 visible_folder,
                 infrared_folder,
                 mask_folder, 
                 label_file,
                 mode = "train" # train | valid | test
                ):
        
        _label_df = pd.read_csv(label_file)
        self.label_df        = _label_df[_label_df["mode"] == mode]        
        self.mode            = mode
        self.visible_folder  = visible_folder
        self.infrared_folder = infrared_folder
        self.mask_folder     = mask_folder    
        self.augment         = Augment(mode)
        self.augment2        = Augment('valid')
        self.mask_dict       = {"plantation"             : 1,
                                "grassland shrubland"    : 2,
                                "smallholder agriculture": 3,
                                "other"                  : 4}
        
    def __len__(self):        
        return len(self.label_df)
        
    def __getitem__(self, index):
                
        case_id, deforestation_type, lat, long, year, _ = self.label_df.iloc[index].to_list()
        
        # load image and mask
        visible  = cv2.imread(self.visible_folder  + str(case_id) + "/composite.png")
        infrared = cv2.imread(self.infrared_folder + str(case_id) + "/composite.png")
        mask     = cv2.imread(self.mask_folder     + str(case_id) + ".png", 0) if (self.mode != "test") else np.zeros(visible.shape[:2]) # dummy mask for test-set case.
        
        # convert the foreground region in the mask to the corressponding label integer
        label = self.mask_dict[deforestation_type]
        mask[mask == 1.] = label
        
        #augment mask and image
        
        if CFG.use_vi_inf:
            visible, infrared, mask = self.augment(image  = visible,
                                                image2 = infrared,
                                                mask   = mask).values()
            image = np.concatenate((visible, infrared), axis = -1)
        else:
            visible, mask = self.augment(image  = visible,
                                                mask   = mask).values()
            image = visible

        # if deforestation_type == 'grassland shrubland' or deforestation_type == 'other':
        # else:
        #     visible, infrared, mask = self.augment2(image  = visible,
        #                                         image2 = infrared,
        #                                         mask   = mask).values()
        
        
        
        return torch.tensor(image), torch.tensor(mask), label, case_id

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

train_dataset = FOREST(visible_folder, infrared_folder, mask_folder, label_file,
                       mode = "train")

for i in range(600,605):
    image, mask, *_ = train_dataset[i]
    
    visible = image[..., :3]
    
    # show_image(visible, mask = mask)
    # plt.show()

# %%
# load models
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_channels = 3+3 if CFG.use_vi_inf else 3

if CFG.seg_model_name == "UNet":
    model = smp.Unet(encoder_name    = CFG.encoder_name,
                    encoder_weights = "imagenet",
                    in_channels     = num_channels,
                    classes         = CFG.num_class+1).to(CFG.device)
elif CFG.seg_model_name == "UNetPlusPlus":
    model = smp.UnetPlusPlus(
            encoder_name=CFG.encoder_name,      
            encoder_weights="imagenet",     
            in_channels=num_channels,     
            classes=CFG.num_class+1,).to(CFG.device)
elif CFG.seg_model_name == "UIUNet":
    model = model = UIUNET(in_ch=num_channels, out_ch=CFG.num_class+1).to(CFG.device)

print(count_parameters(model))

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
    pred = (torch.argmax(pred, dim = 1) == label).long().view(-1)        
    mask = mask.view(-1)

    # compute hard dice score for the foreground region
    score = (torch.sum(pred * mask)*2)/ (torch.sum(pred) + torch.sum(mask) + eps)    
    
    return np.array(score)

alpha = 0.3
beta = 1 - alpha
TverskyLoss = smp.losses.TverskyLoss(mode='multiclass', log_loss=False, alpha=alpha, beta=beta)
DiceLoss    = smp.losses.DiceLoss(mode='multiclass')
CELoss     = smp.losses.SoftCrossEntropyLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='multiclass', per_image=False)

# %%
loss_fn = TverskyLoss
CFG.init_lr = 0.0005
# optimizer = optim.Adam(model.parameters(), lr=CFG.init_lr)
optimizer = optim.AdamW(model.parameters(), lr=CFG.init_lr)
# learning rate scheduler
scheduler = get_scheduler(CFG, optimizer)

# %%
def train(trainloader, validloader, model,
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
                torch.save(model.state_dict(), f"./{valid_dice:.3f}_{CFG.save_weight_path}")
        
    return model

def train_epoch(trainloader, model):
        
    losses = []
    iters = len(trainloader)
    
    for (inputs, targets, *_) in trainloader:
        # forward pass
        if CFG.seg_model_name == 'UIUNet':
            d0, d1, d2, d3, d4, d5, d6 = model(inputs.permute(0,-1,1,2).to(CFG.device))
        else:
            outputs = model(inputs.permute(0,-1,1,2).to(CFG.device)) # channel first
        targets = targets.long().to(CFG.device)
        # calculate loss
        if CFG.seg_model_name == 'UIUNet':
            loss0 = loss_fn(d0, targets)
            # loss1 = loss_fn(d1, targets)
            # loss2 = loss_fn(d2, targets)
            # loss3 = loss_fn(d3, targets)
            # loss4 = loss_fn(d4, targets)
            # loss5 = loss_fn(d5, targets)
            # loss6 = loss_fn(d6, targets)
            # loss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/7
            loss = loss0
        else:
            loss = loss_fn(outputs, targets)

        # backward pass and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        match CFG.scheduler:
            case 'ReduceLROnPlateau':
                scheduler.step(loss) # 
            case 'CosineAnnealingLR': # CosineAnnealingWarmRestarts
                scheduler.step()


        losses.append(loss.item())
    
    return np.mean(losses)

def evaluate_epoch(validloader, model):
    model.eval()
    scores = []
    loss = []
    for (inputs, targets, label, _) in validloader:
        if CFG.seg_model_name == 'UIUNet':
            d0, d1, d2, d3, d4, d5, d6 = model(inputs.permute(0,-1,1,2).to(CFG.device))
            d0 = d0.detach().cpu()
            # d1 = d1.detach().cpu()
            # d2 = d2.detach().cpu()
            # d3 = d3.detach().cpu()
            # d4 = d4.detach().cpu()
            # d5 = d5.detach().cpu()
            # d6 = d6.detach().cpu()
            outputs = d0
        else:
            outputs = model(inputs.permute(0,-1,1,2).to(CFG.device)).detach().cpu() #channel first
        targets = targets.long()
        # calculate loss
        if CFG.seg_model_name == 'UIUNet':
            loss0 = loss_fn(d0, targets)
            # loss1 = loss_fn(d1, targets)
            # loss2 = loss_fn(d2, targets)
            # loss3 = loss_fn(d3, targets)
            # loss4 = loss_fn(d4, targets)
            # loss5 = loss_fn(d5, targets)
            # loss6 = loss_fn(d6, targets)
            # val_loss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/7
            val_loss = loss0
        else:
            val_loss = loss_fn(outputs, targets)
        #calculate dice
        score = hard_dice(outputs, targets, label)
        
        loss.append(val_loss.item())
        scores.append(score)
    
    return np.mean(loss), np.mean(scores)

# %%
visible_folder  = "./dataset/processed/visibles/"
infrared_folder = "./dataset/processed/infrareds/"
mask_folder     = "./dataset/processed/masks/"
label_file      = "./dataset/processed/label.csv"

train_dataset = FOREST(visible_folder, infrared_folder, mask_folder, label_file,
                       mode = "train")
valid_dataset = FOREST(visible_folder, infrared_folder, mask_folder, label_file,
                       mode = "valid")

train_loader = DataLoader(train_dataset,
                          batch_size  = CFG.batch_size,
                          num_workers = 14,
                          shuffle     = True, 
                          pin_memory  = True)

valid_loader = DataLoader(train_dataset,
                          batch_size  = 1,
                          num_workers = 8,
                          shuffle     = False,
                          pin_memory  = False)

# %%
# Train k-Fold
# Split your dataset into K-folds
# kf = KFold(n_splits=CFG.n_fold, shuffle=True)
# for fold, (train_idx, val_idx) in enumerate(kf.split(visible_folder)):
#     if fold != CFG.train_fold:
#         continue
#     train_data = visible_folder[train_idx]
#     train_dataset = FOREST(train_data1, train_data2, mask_folder, label_file, mode='train')
#     train_loader = DataLoader(train_dataset, 
#                                   batch_size=CFG.batch_size, 
#                                   num_workers=14, 
#                                   shuffle=True, 
#                                   pin_memory=True)

#     # Create a PyTorch DataLoader for the validation set
#     val_data = visible_folder[val_idx]
#     val_dataset = FOREST(val_data1, val_data2, mask_folder, label_file, mode='valid')
#     val_loader = DataLoader(val_dataset, 
#                                 batch_size=1, 
#                                 num_workers=8, 
#                                 shuffle=False,
#                                 pin_memory=False)
#     model = train(train_loader, valid_loader, model,
#               n_epoch = CFG.epochs)
# %%
# Train
model = train(train_loader, valid_loader, model,
              n_epoch = CFG.epochs)

# %%
# Submission
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

# %%
visible_folder  = "./dataset/processed/visibles/"
infrared_folder = "./dataset/processed/infrareds/"
mask_folder     = "./dataset/processed/masks/"
label_file      = "./dataset/processed/label.csv"

test_dataset = FOREST(visible_folder, infrared_folder, mask_folder, label_file,
                      mode = "test")

test_loader = DataLoader(test_dataset,
                         batch_size  = 1,
                         num_workers = 14,
                         shuffle     = False,
                         pin_memory  = False)

# %%
# load model
# model.load_state_dict(torch.load("./0.310_weights_dice_resnet101.pth"))

# test_results = predict(model, test_loader)

# df_submission = pd.DataFrame.from_dict(test_results)

# df_submission.to_csv("my_submission.csv", index = False)
# %%
