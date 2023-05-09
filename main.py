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

import segmentation_models_pytorch as smp
import albumentations as A
import timm

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler

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
    seg_model_name = 'UNet' # UNetPlusPlus

    ensemble       = False
    img_size       = 320
    scheduler      = "CosineAnnealingLR" #"CosineAnnealingLR" #"ReduceLROnPlateau" #'CosineAnnealingWarmRestarts'
    epochs         = 10
    init_lr        = 0.0005
    min_lr         = 1e-6
    T_0            = 25
    batch_size     = 16
    weight_decay   = 1e-6
    
    seed           = 42
    n_fold         = 4
    trn_fold       = [0]

    num_class      = 4
    save_weight_path     =  f'weights_dice_{encoder_name}.pth'

    device         = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

set_seed(CFG.seed)
# %%
def Augment(mode):
    if mode == "train":
    
        return A.Compose([# A.Resize(CFG.img_size, CFG.img_size),
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
                          ],)
                         #additional_targets={'image2': 'image'}) # this is to augment both the normal and infrared sattellite images.
    
    else: # valid test
        return A.Compose([# A.Resize(CFG.img_size, CFG.img_size), 
                          A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))],
                         )#additional_targets={'image2': 'image'})

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
        # if deforestation_type == 'grassland shrubland' or deforestation_type == 'other':
        visible, mask = self.augment(image  = visible,
                                            # image2 = infrared,
                                            mask   = mask).values()
        # else:
        #     visible, infrared, mask = self.augment2(image  = visible,
        #                                         image2 = infrared,
        #                                         mask   = mask).values()
        
        # concat visible and infared and a single 5-channel image
        # image = np.concatenate((visible, infrared), axis = -1)
        image = visible
        
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
    
    show_image(visible, mask = mask)
    plt.show()

# %%
# ResNet Attention

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, is_deconv=False, scale=True):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        nonlinearity = nn.ReLU
        self.relu1 = nonlinearity(inplace=True)

        if scale:
            # B, C/4, H, W -> B, C/4, H, W
            if is_deconv:
                self.upscale = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                                  stride=2, padding=1, output_padding=1)
            else:
                self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upscale = nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.upscale(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        x = self.conv(x)
        return x



class Resnest(nn.Module):

    def __init__(self, model_name, out_neurons=600):
        super().__init__()
        try:
            self.backbone = timm.create_model(model_name, pretrained=True)
        except:
            self.backbone = torch.hub.load('zhanghang1989/ResNeSt', model_name, pretrained=True)
            
        self.in_features = 2048
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        try:
            x = self.backbone.act1(x)
        except:
            x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        layer1 = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)

        return x, layer1, layer2, layer3

class ResnestDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        nonlinearity = nn.ReLU
        self.decode1 = DecoderBlock(1024, 512)
        self.decode2 = DecoderBlock(512, 256)
        self.decode3 = DecoderBlock(256, 64)
        self.decode4 = DecoderBlock(64, 32)
        self.decode5 = DecoderBlock(32, 16)
        self.conv1 = conv_block(1024, 512)
        self.conv2 = conv_block(512, 256)
        self.conv3 = conv_block(128, 64)
        self.Att1 = Attention_block(512, 512, 256)
        self.Att2 = Attention_block(256, 256, 64)
        self.Att3 = Attention_block(64, 64, 32)
        self.Att4 = Attention_block(64, 64, 32)
        self.conv4 = nn.Conv2d(64, 64, 3, 2, 1)
        self.finalconv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(4, 1, 3, padding=1)
    
    def forward(self, x, l1, l2, l3):
        d1 = self.decode1(l3)
        l2 = self.Att1(d1, l2)
        d1 = torch.cat((l2,d1),dim=1)
        d1 = self.conv1(d1)
        d2 = self.decode2(d1)
        l1 = self.Att2(d2, l1)
        d2 = torch.cat((l1,d2),dim=1)
        d2 = self.conv2(d2)
        d3 = self.decode3(d2)
        d3 = self.conv4(d3)
        x = self.Att3(d3, x)
        d3 = torch.cat((x,d3),dim=1)
        d3 = self.conv3(d3)
        d4 = self.decode4(d3)
        d5 = self.decode5(d4)
        out = self.finalconv2(d5)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

class resUnest(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()
        self.resnest = Resnest(model_name=encoder_model)
        self.decoder = ResnestDecoder()
        
    def forward(self, x):
        x, l1, l2, l3 = self.resnest(x)
        out = self.decoder(x, l1, l2, l3)
        return out
# %%
# load models
if CFG.seg_model_name == "UNet":
    model = smp.Unet(encoder_name    = CFG.encoder_name,
                    encoder_weights = "imagenet",
                    in_channels     = 3,
                    classes         = CFG.num_class+1).to(CFG.device)
elif CFG.seg_model_name == "UNetPlusPlus":
    model = smp.UnetPlusPlus(
            encoder_name=CFG.encoder_name,      
            encoder_weights="imagenet",     
            in_channels=3+3,                  
            classes=CFG.num_class+1,).to(CFG.device)
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
# def criterion(y_pred, y_true):
#     return 0.5*CELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)
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
                torch.save(model.state_dict(), f"./{CFG.save_weight_path}")
        
    return model

def train_epoch(trainloader, model):
        
    losses = []
    iters = len(trainloader)
    
    for (inputs, targets, *_) in trainloader:
        # forward pass       
        outputs = model(inputs.permute(0,-1,1,2).to(CFG.device)) # channel first
        targets = targets.long().to(CFG.device)

        # calculate loss
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
    
    scores = []
    loss = []
    for (inputs, targets, label, _) in validloader:
        
        outputs = model(inputs.permute(0,-1,1,2).to(CFG.device)).detach().cpu() #channel first
        targets = targets.long()
        
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

# load model
model.load_state_dict(torch.load("./weights_dice_resnet101.pth"))

test_results = predict(model, test_loader)

df_submission = pd.DataFrame.from_dict(test_results)

df_submission.to_csv("my_submission.csv", index = False)
# %%
