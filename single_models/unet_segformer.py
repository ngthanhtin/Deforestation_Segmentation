from Indonesia_Deforestation_Segmentation.single_models.unet import UNET
from torch import nn
import torch

from transformers import SegformerForSemanticSegmentation

class UNET_Segformer(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout)

        self.encoder = UNET(in_channels=in_channels, out_channels=num_classes)

        self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                                           num_labels=num_classes,
                                                                           ignore_mismatched_sizes=True,
                                                                           num_channels=num_classes)
        
        self.upscaler_1 = nn.ConvTranspose2d(num_classes,1,kernel_size=(4,4), stride=2, padding=1)
        self.upscaler_2 = nn.ConvTranspose2d(1,num_classes,kernel_size=(4,4), stride=2, padding=1)

    def forward(self, x):
        output = self.encoder(x)
        output = self.dropout(output)
        
        output = self.encoder_2d(output).logits
        output = self.upscaler_1(output)
        output = self.upscaler_2(output)

        return output
    
def test():
    x = torch.randn((3, 6, 320, 320))
    model = UNET_Segformer(in_channels=6, num_classes=4+1)
    preds = model(x)

                                                                           