from single_models.unetr import UNETR
from torch import nn
import torch

from transformers import SegformerForSemanticSegmentation

class UNETR_Segformer(nn.Module):
    def __init__(self, cfg, dropout=0.2):
        super().__init__()
        self.cfg = cfg
        self.dropout = nn.Dropout2d(dropout)

        self.encoder = UNETR(input_dim=1, output_dim=32, img_shape=(3, 320, 320))

        self.encoder_2d = SegFormerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                                           num_labels=1,
                                                                           ignore_mismatched_sizes=True,
                                                                           num_channels=32)
        
        self.upscaler_1 = nn.ConvTranspose2d(1,1,kernel_size=(4,4), stride=2, padding=1)
        self.upscaler_2 = nn.ConvTranspose2d(1,1,kernel_size=(4,4), stride=2, padding=1)

    def forward(self, image):
        output = self.encoder(image).max(axis=2)[0]
        output = self.dropout(output)
        output = self.encoder_2d(output).logits
        output = self.upscaler_1(output)
        output = self.upscaler_2(output)

        return output
                                                                           