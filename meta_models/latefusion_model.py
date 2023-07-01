import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
# import segmentation_models_pytorch as smp

sigmoid = nn.Sigmoid()


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class Meta_Seg(nn.Module):
    def __init__(self, seg_model, in_channels, n_labels, n_meta_features=0, n_meta_dim=[32, 16]):
        super(Meta_Seg, self).__init__()
        self.n_meta_features = n_meta_features
        
        self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                # nn.Dropout(p=0.2),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
        )
        
            
        self.meta_fc = nn.Linear(n_meta_dim[1], n_labels)

        self.seg_model = seg_model
        
    def extract(self, x):
        return self.seg_model(x)

    def forward(self, x, x_meta):
        x = self.extract(x) # B, C, H, W
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x_meta = self.meta_fc(x_meta)
            x_meta  = x_meta.unsqueeze(-1).unsqueeze(-1)
            
            x = x+x_meta
    
        return x