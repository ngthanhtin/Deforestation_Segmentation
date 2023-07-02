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

from PIL import Image

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from tqdm import tqdm

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
# Config
class CFG:
    visible_folder  = "./dataset/processed/visibles/"
    infrared_folder = "./dataset/processed/infrareds/"
    mask_folder     = "./dataset/processed/masks/"
    label_file      = "./dataset/processed/label_remove_small_pixels.csv"

    encoder_name   = 'tu-eca_nfnet_l1' # resnet101, efficientnet-b6, timm-regnety_008, timm-regnety_120
    seg_model_name = 'segformer' # segformer, UNetPlusPlus, UIUNet, UNet, PAN, NestedUNet, DeepLabV3Plus
    activation     = None #softmax2d, sigmoid, softmax

    ensemble       = False
    ensemble_model_names = ['segformer', 'UNetPlusPlus']
    ensemble_model_paths = ['./results/segformer_weights_06_29_2023-10:49:17/-1_0.342_weights_segformer_2_images_False_meta.pth',\
                # 'results/UNetPlusPlus_weights_06_29_2023-10:31:25/-1_0.330_weights_UNetPlusPlus_2_images_False_meta.pth']
                './results/0.261_0.358_weights_dice_resnet101_UNetPlusPlus_2images.pth']

    use_vi_inf     = True
    img_size       = 320

    batch_size     = 8
    
    seed           = 42

    num_class      = 4 # 4
    num_inputs     = 2 if use_vi_inf else 1
    use_meta       = False

    load_weight_folder = 'results/segformer_weights_07_02_2023-06:57:15/'
    specific_weight_file = '-1_0.332_weights_segformer_2_images_False_meta.pth'
    device         = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    submission     = True
    visualize      = False
    eval           = False
    TTA            = False

set_seed(CFG.seed)

preprocessing_fn = lambda image : get_preprocessing_fn(encoder_name = CFG.encoder_name,
                                                       pretrained = 'imagenet')
preprocessing_fn = None
# %%

def Augment(mode):
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
        
        return torch.tensor(image), torch.tensor(mask), label, case_id

# %%
# load models
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_channels = 3+3 if CFG.use_vi_inf else 3
print(f"Number of channels: {num_channels}")

def build_model(CFG, model_name):
    if model_name == 'segformer':
        #model settings
        from importlib import import_module
        module = import_module(f'mmseg.utils')
        module.register_all_modules(True)

        norm_cfg = dict(type='BN', requires_grad=True)
        model_cfg = dict(
            type='EncoderDecoder',
            pretrained=None,
            backbone=dict(
                type='MixVisionTransformer',
                in_channels=num_channels,
                embed_dims=64,
                num_stages=4,
                num_layers=[3, 8, 27, 3],
                num_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1,
                init_cfg = dict(type="Pretrained", checkpoint="segformer_checkpoints/mit_b5_mmseg.pth")),
            decode_head=dict(
                type='SegformerHead',
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                channels=256,
                dropout_ratio=0.1,
                num_classes=CFG.num_class + 1,
                norm_cfg=norm_cfg,
                align_corners=False,
                # loss_decode=dict(type='DiceLoss', use_sigmoid=False, loss_weight=1.0)),
                loss_decode=[
                    dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                    dict(type='DiceLoss', use_sigmoid=False, loss_weight=3.0)]),
            # model training and testing settings
            train_cfg=dict(),
            test_cfg=dict(mode='whole'))

        from mmseg.models import build_segmentor
        model = build_segmentor(model_cfg).to(CFG.device)
        model.init_weights()

    elif model_name == "UNet":
        model = smp.Unet(encoder_name    = CFG.encoder_name,
                        encoder_weights = "imagenet",
                        in_channels     = num_channels,
                        classes         = CFG.num_class+1,
                        activation=CFG.activation).to(CFG.device)
    elif model_name == "UNetPlusPlus":
        model = smp.UnetPlusPlus(
                encoder_name=CFG.encoder_name,      
                encoder_weights="ssl",
                in_channels=num_channels,     
                classes=CFG.num_class+1,
                activation=CFG.activation).to(CFG.device)
    
    return model

# %% test time augmentation
import albumentations as A

identity_trfm = A.Lambda(image = lambda x,cols=None,rows=None : x)
# Affine transforms
horizontal_flip = A.HorizontalFlip(p = 1.0)
vertical_flip = A.VerticalFlip(p = 1.0)
rotate_cw = A.Rotate(limit = (-90, -90), p = 1.0)
rotate_acw = A.Rotate(limit = (90, 90), p = 1.0)

# Pixel level transformations
pixel_level_trfms = A.OneOf([
                    A.HueSaturationValue(10,15,10),
                    # A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(),            
                   ], p = 1.0)

# List of augmentations for TTA
tta_augs = [identity_trfm,
            horizontal_flip,
            vertical_flip,
            rotate_cw, pixel_level_trfms]

# List of deaugmentations corresponding to the above aug list
tta_deaugs = [None,
              horizontal_flip,
              vertical_flip,
              rotate_acw, None]
# %%
class EnsembleModel(nn.Module):
    def __init__(self, model_names, model_paths):
        super().__init__()
        self.models = nn.ModuleList()
        self.model_names = model_names
        self.model_paths = model_paths
        for model_name, model_path in zip(model_names, model_paths):
            model = build_model(CFG, model_name)
            model.to(CFG.device)

            model.load_state_dict(torch.load(model_path))
            model.eval()

            self.models.append(model)
    
    def forward(self, x):
        output=[]
        for m, model_name in zip(self.models, self.model_names):
            if CFG.TTA and model_name != 'segformer':
                tta_pred = None
                inputs = x.permute(2,3,1,0).squeeze()
                inputs = inputs.detach().cpu().numpy()
                
                for j, tta_aug in enumerate(tta_augs):
                    # Augmentation    
                    aug_img = tta_aug(image = inputs)['image']
                    aug_img = torch.tensor(aug_img)
                    aug_img = aug_img.unsqueeze(0)

                    with torch.no_grad():
                        outputs = m(aug_img.permute(0,-1,1,2).to(CFG.device)).detach().cpu() #channel first
                        if CFG.seg_model_name == 'segformer' and not CFG.ensemble:        
                            outputs = F.interpolate(outputs, (320, 320), mode = 'bilinear')
                        # Deaugmentation
                        outputs = outputs.permute(2,3,1,0).squeeze()
                        outputs = outputs.detach().cpu().numpy()
                        if tta_deaugs[j] is not None:
                            outputs = tta_deaugs[j](image = inputs, 
                                                mask = outputs)['mask']          

                        if tta_pred is None:
                            tta_pred = outputs
                        else:       
                            tta_pred += outputs
                output_ = tta_pred / len(tta_augs)
                output_ = torch.tensor(output_).unsqueeze(0).permute(0,-1,1,2).to(CFG.device)
            else:
                output_ = m(x)
                if model_name == 'segformer':
                    output_ = F.interpolate(output_, (320, 320), mode = 'bilinear')
            output.append(output_)
        output=torch.stack(output,dim=0).mean(0)

        return output

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

# %%
def evaluate_epoch(validloader, model):
    model.eval()
    scores = []
    
    for (inputs, targets, label, _) in tqdm(validloader):
        if CFG.TTA and not CFG.ensemble:
            tta_pred = None
            inputs = inputs.squeeze()
            inputs = inputs.detach().cpu().numpy()
            
            for j, tta_aug in enumerate(tta_augs):
                # Augmentation    
                if j == len(tta_augs) - 1:
                    aug_img1 = tta_aug(image = inputs[:, :, :3])['image']
                    aug_img2 = tta_aug(image = inputs[:, :, 3:])['image']
                    aug_img = np.concatenate((aug_img1, aug_img2), axis = -1)
                else:
                    aug_img = tta_aug(image = inputs)['image']
                aug_img = torch.tensor(aug_img)
                aug_img = aug_img.unsqueeze(0)

                with torch.no_grad():
                    outputs = model.forward(aug_img.permute(0,-1,1,2).to(CFG.device)).detach().cpu() #channel first
                    if CFG.seg_model_name == 'segformer' and not CFG.ensemble:        
                        outputs = F.interpolate(outputs, (320, 320), mode = 'bilinear')
                    # Deaugmentation
                    outputs = outputs.permute(2,3,1,0).squeeze()
                    outputs = outputs.detach().cpu().numpy()
                    if tta_deaugs[j] is not None:
                        outputs = tta_deaugs[j](image = inputs, 
                                              mask = outputs)['mask']          

                    if tta_pred is None:
                        tta_pred = outputs
                    else:       
                        tta_pred += outputs
            outputs = tta_pred / len(tta_augs)
            outputs = torch.tensor(outputs).unsqueeze(0).permute(0,-1,1,2)

        else:
            outputs = model.forward(inputs.permute(0,-1,1,2).to(CFG.device)).detach().cpu() #channel first
            if CFG.seg_model_name == 'segformer' and not CFG.ensemble:        
                outputs = F.interpolate(outputs, (320, 320), mode = 'bilinear')
        targets = targets.long()

        
        #calculate dice
        score = hard_dice(outputs, targets, label)
        
        scores.append(score)
    
    return np.mean(scores)

# %%
# Test on real validation set
label_file      = "./dataset/processed/label.csv"
label_df        = pd.read_csv(label_file)
label_df['data_folder'] = ['./dataset']*len(label_df)

val_df = label_df[label_df['mode'] == 'valid']
test_df = label_df[label_df['mode'] == 'test']

valid_dataset = FOREST(val_df, mode = "valid")
test_dataset = FOREST(test_df, mode = "test")

print(f"Len of full valid dataset: {len(valid_dataset)}")
print(f"Len of full test dataset: {len(test_dataset)}")

valid_loader = DataLoader(valid_dataset,
                          batch_size  = 1,
                          num_workers = 8,
                          shuffle     = False,
                          pin_memory  = False)

test_loader = DataLoader(test_dataset,
                         batch_size  = 1,
                         num_workers = 14,
                         shuffle     = False,
                         pin_memory  = False)
# %%
weight_paths = os.listdir(CFG.load_weight_folder)
if CFG.specific_weight_file:
    weight_paths = [CFG.specific_weight_file]
weight_paths.sort(key=lambda x: x[0])
weight_paths = [os.path.join(CFG.load_weight_folder, p) for p in weight_paths]

if not CFG.ensemble:
    model = build_model(CFG, CFG.seg_model_name)
    if CFG.eval:
        for path in weight_paths:
            model.load_state_dict(torch.load(path))
            print(f"Inference model: {path}")
            model.eval()
            with torch.no_grad():
                valid_dice = evaluate_epoch(valid_loader, model)
                print(f"Valid Dice: {valid_dice}, LB Score: {valid_dice-0.1}")
else:
    model = EnsembleModel(CFG.ensemble_model_names, CFG.ensemble_model_paths)
    if CFG.eval:
        print(f"Inference ensemble models: ", CFG.ensemble_model_paths)
        model.eval()
        with torch.no_grad():
            valid_dice = evaluate_epoch(valid_loader, model)
            print(f"Valid Dice: {valid_dice}, LB Score: {valid_dice-0.1}")
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
    for (inputs, _, label, image_id) in tqdm(loader):
        if CFG.TTA:
            tta_pred = None
            inputs = inputs.squeeze()
            inputs = inputs.detach().cpu().numpy()
            
            for j, tta_aug in enumerate(tta_augs):
                # Augmentation    
                if j == len(tta_augs) - 1:
                    aug_img1 = tta_aug(image = inputs[:, :, :3])['image']
                    aug_img2 = tta_aug(image = inputs[:, :, 3:])['image']
                    aug_img = np.concatenate((aug_img1, aug_img2), axis = -1)
                else:
                    aug_img = tta_aug(image = inputs)['image']
                
                aug_img = torch.tensor(aug_img)
                aug_img = aug_img.unsqueeze(0)

                with torch.no_grad():
                    outputs = model.forward(aug_img.permute(0,-1,1,2).to(CFG.device)).detach().cpu() #channel first
                    if CFG.seg_model_name == 'segformer' and not CFG.ensemble:        
                        outputs = F.interpolate(outputs, (320, 320), mode = 'bilinear')
                    # Deaugmentation
                    outputs = outputs.permute(2,3,1,0).squeeze()
                    outputs = outputs.detach().cpu().numpy()
                    if tta_deaugs[j] is not None:
                        outputs = tta_deaugs[j](image = inputs, 
                                              mask = outputs)['mask']          

                    if tta_pred is None:
                        tta_pred = outputs
                    else:       
                        tta_pred += outputs
            pred = tta_pred / len(tta_augs)
            pred = torch.tensor(pred).unsqueeze(0).permute(0,-1,1,2)
        else:
            # forward pass       
            pred = model(inputs.permute(0,-1,1,2).to(CFG.device)) # channel first
            if CFG.seg_model_name == 'segformer' and not CFG.ensemble:
                pred = F.interpolate(pred, (320, 320), mode = 'bilinear')
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

# %% visualization
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

if CFG.visualize:
    if not CFG.ensemble:
        model.load_state_dict(torch.load(f"{CFG.load_weight_folder}/{CFG.specific_weight_file}"))
    model.eval()

# %%
import random
random_ids = [random.randint(0, 357) for _ in range(10)]
if CFG.visualize:
    for i in random_ids:
        image, mask, label, case_id = valid_dataset[i]
        visible = image[..., :3]
        
        mask_predict = model(image.unsqueeze(0).permute(0,-1,1,2).to(CFG.device))
        mask_predict = F.interpolate(mask_predict, (320, 320)).cpu()
        print("Dice: ", hard_dice(mask_predict, mask, label))

        predict_class = torch.argmax(mask_predict, dim = 1)
        mask_predict = (torch.argmax(mask_predict, dim = 1) == label).squeeze(0).long().numpy()
        mask_predict *= label

        if CFG.use_vi_inf:
            print("GT: ", case_id, label, torch.mean(mask.float()))
            show_image(visible, mask = mask)
            plt.show()
            print("Predict: ", np.mean(mask_predict))
            show_image(visible, mask = mask_predict)
            plt.show()
        else:
            show_image(visible, mask = mask[0])
            plt.show()
        


# %%
# load model
if CFG.submission:
    if not CFG.ensemble:
        model.load_state_dict(torch.load(f'{CFG.load_weight_folder}/{CFG.specific_weight_file}'))

    test_results = predict(model, test_loader)

    df_submission = pd.DataFrame.from_dict(test_results)

    df_submission.to_csv(f"my_submission.csv", index = False)
# %%
