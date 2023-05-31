# %%
# MODEL
import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision import transforms

from segment_anything import sam_model_registry

import torch.nn as nn
import torch

# %%
class SAMFinetuner(nn.Module):
    def __init__(
            self,
            model_type,
            checkpoint_path,
            freeze_image_encoder=False,
            freeze_prompt_encoder=False,
            freeze_mask_decoder=False,
            device='cpu'
        ):
        super(SAMFinetuner, self).__init__()

        self.model_type = model_type
        self.model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)

        self.model.to(device=device)
        self.freeze_image_encoder = freeze_image_encoder
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False
        
        print(self.model.mask_decoder.num_mask_tokens)

    def forward(self, imgs):
        _, B, H, W = imgs.shape
        features = self.model.image_encoder(imgs)
        num_masks = B
#         num_masks = sum([len(b) for b in bboxes])
        predictions = []
        for feature in features:
            # Embed prompts
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            # Predict masks
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=feature.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # Upscale the masks to the original image resolution
            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )

            predictions.append([masks, iou_predictions])
            # masks = masks.squeeze(1).flatten(1)

        return predictions
        
# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = 'cuda:7'
model = SAMFinetuner(
    "vit_b",
    "./sam_vit_b_01ec64.pth",
    freeze_image_encoder=True,
    freeze_prompt_encoder=True,
    freeze_mask_decoder=False,
    device=device
)
print(count_parameters(model))
model.eval()
x = torch.randn((4, 3, 1024, 1024)).to(device) # Input of SAM must be (Batch,3,1024,1024)
out1 = model(x)[0][0] #output mask shape: (Batch,1,1,1024,1024)

y = torch.randn((4, 3, 1024, 1024)).to(device)
out2 = model(y)[0][0]

cmb = out1+out2
print(cmb.shape)
print(torch.sigmoid(cmb))
# %%
