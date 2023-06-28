import torch
import segmentation_models_pytorch as smp


#segformer model settings
from importlib import import_module
module = import_module(f'mmseg.utils')
module.register_all_modules(True)

def build_segmentation_model(in_channels, num_classes, model_name, encoder_name):
    if model_name == 'segformer':
        norm_cfg = dict(type='BN', requires_grad=True)
        model_cfg = dict(
            type='EncoderDecoder',
            # data_preprocessor=dict(
            # type='SegDataPreProcessor',
            # bgr_to_rgb=True,
            # pad_val=0,
            # seg_pad_val=0),
            pretrained=None,
            backbone=dict(
                type='MixVisionTransformer',
                in_channels=in_channels,
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
                num_classes=num_classes + 1,
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
        model = build_segmentor(model_cfg)
        model.init_weights()

    elif model_name == "UNet":
        model = smp.Unet(encoder_name    = encoder_name,
                        encoder_weights = "imagenet",
                        in_channels     = in_channels,
                        classes         = num_classes+1,)
    elif model_name == "UNetPlusPlus":
        model = smp.UnetPlusPlus(
                encoder_name=encoder_name,      
                encoder_weights="imagenet",
                in_channels=in_channels,     
                classes=num_classes+1,)
    
    return model

def build_meta_model():
    model = 1
    return model



class LateFusion_Model(torch.nn.Module):
    def __init__(self, in_channel=1, num_classes=1, model_name='segformer', encoder_name='mixvisiontransformer',\
                  n_metadata=None):
        super().__init__()

        self.seg_model = build_segmentation_model(in_channels=in_channel, num_classes=num_classes+1, model_name=model_name, encoder_name=encoder_name)
        self.meta_model = build_meta_model()

    def forward(self, x, context):
        masks = self.seg_model(x)
        masks *= self.meta_model(context)

        return masks