import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn import Parameter, Softmax


class Img_Input(nn.Module):
    '''(conv => BN => ReLU) '''
    def __init__(self, in_ch, out_ch):
        super(Img_Input, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class Img_Encode_Block(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(Img_Encode_Block, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.maxpool(x)
        residual = x
        x = self.conv(x)
        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x += residual
        return x


class Inf_Input(nn.Module):
    '''(conv => BN => ReLU) '''

    def __init__(self, in_ch, out_ch):
        super(Inf_Input, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Inf_Encode_Att(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool_h = nn.AdaptiveMaxPool2d((None,1))
        self.avgpool_h = nn.AdaptiveAvgPool2d((None,1))
        
        self.maxpool_w = nn.AdaptiveMaxPool2d((1, None))
        self.avgpool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, channel // reduction)
        self.cat_conv = nn.Sequential(
            nn.Conv2d(channel, mip, 1, bias=False),
            nn.BatchNorm2d(mip),
            nn.ReLU()
        )
        
        self.conv_h = nn.Conv2d(mip,channel,kernel_size=1,stride=1,padding=0)
        self.conv_w = nn.Conv2d(mip,channel,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        h_max = self.maxpool_h(x)
        w_max = self.maxpool_w(x).permute(0, 1, 3, 2)
        
        h_avg = self.avgpool_h(x)
        w_avg = self.avgpool_w(x).permute(0, 1, 3, 2)
        
        avg_cat = torch.cat([h_avg, w_avg], dim=2)
        avg_cat = self.cat_conv(avg_cat)
        h_avg, w_avg = torch.split(avg_cat, [h, w], dim=2)
        w_avg = w_avg.permute(0,1,3,2)
        max_cat = torch.cat([h_max, w_max], dim=2)
        max_cat = self.cat_conv(max_cat)
        h_max, w_max = torch.split(max_cat, [h, w], dim=2)
        w_max = w_max.permute(0,1,3,2)
        
        h_att = self.conv_h(h_max+h_avg)
        w_att = self.conv_w(w_max+w_avg)
        h_att = self.sigmoid(h_att)
        w_att = self.sigmoid(w_att)
        h_att = h_att.expand(-1, -1, h, w)
        w_att = w_att.expand(-1, -1, h, w)
        output = identity * w_att * h_att
        return output


class Inf_Encode_Block(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(Inf_Encode_Block, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        

    def forward(self, x):
        x = self.maxpool(x)
        residual = x
        x = self.conv(x)
        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x += residual
        return x


class InfPamAtt(nn.Module):
    ''' Position attention module'''
    def __init__(self, x_dim, inf_dim):
        super(InfPamAtt, self).__init__()
        self.chanel_in = inf_dim

        self.query_conv = nn.Conv2d(
            in_channels=inf_dim, out_channels=inf_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=inf_dim, out_channels=inf_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=x_dim, out_channels=x_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    
    def forward(self, x , inf):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(inf).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(inf).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out


class CamAtt(nn.Module):
    ''' Channel attention module'''

    def __init__(self, in_dim):
        super(CamAtt, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class Decode_Block(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(Decode_Block, self).__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        residual = x
        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x += residual
        x = self.upsample(x)
        return x


class ResConcate(nn.Module):
    def __init__(self, cat_ch, img_ch, inf_ch):
        super(ResConcate, self).__init__()
        self.cat_conv = nn.Sequential(
            nn.Conv2d(cat_ch*2, cat_ch*2, 1, bias=False),
            nn.BatchNorm2d(cat_ch*2),
            nn.ReLU(),
            nn.Conv2d(cat_ch*2, cat_ch, 1, bias=False),
            nn.BatchNorm2d(cat_ch),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(cat_ch, cat_ch, 1, bias=False),
            nn.BatchNorm2d(cat_ch),
            nn.ReLU()
        )

    def forward(self, x_cat, x_img, x_inf):
        img_inf = torch.cat([x_img, x_inf], dim=1)
        concat = torch.cat([x_cat, img_inf], dim=1)
        concat = self.cat_conv(concat)
        concat = concat+x_cat
        concat = self.conv(concat)
        concat = concat+img_inf
        concat = self.conv(concat)
        return concat


class Seg_Output(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(Seg_Output, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, 3, padding=1),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(),
            nn.Conv2d(in_ch//2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class INFAttNet(nn.Module):
    def __init__(self, n_class, deep_supervision=False):
        super(INFAttNet, self).__init__()
        img_channels = 3
        inf_channels = 3
        
        self.img_input = Img_Input(img_channels, 32)
        self.img_encoder_1 = Img_Encode_Block(32, 64)
        self.img_encoder_2 = Img_Encode_Block(64, 128)
        self.img_encoder_3 = Img_Encode_Block(128, 256)
        
        self.inf_input = Inf_Input(inf_channels, 32)
        self.inf_encoder_1 = Inf_Encode_Block(32, 64)
        self.inf_encoder_2 = Inf_Encode_Block(64, 128)
        self.inf_encoder_3 = Inf_Encode_Block(128, 256)
        
        self.inf_encoder_att_1 = Inf_Encode_Att(64)
        self.inf_encoder_att_2 = Inf_Encode_Att(128)
        self.inf_encoder_att_3 = Inf_Encode_Att(256)

        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.pam_att3 = InfPamAtt(512, 256)
        self.pam_att2 = InfPamAtt(256, 128)
        self.pam_att1 = InfPamAtt(128, 64)
        self.pam_att = InfPamAtt(64, 64)

        self.cam_att3 = CamAtt(512)
        self.cam_att2 = CamAtt(256)
        self.cam_att1 = CamAtt(128)
        self.cam_att = CamAtt(64)
  
        self.decode_block3 = Decode_Block(512, 256)
        self.decode_block2 = Decode_Block(256, 128)
        self.decode_block1 = Decode_Block(128, 64)
        
        self.resconcate3 = ResConcate(512,256,256)
        self.resconcate2 = ResConcate(256,128,128)
        self.resconcate1 = ResConcate(128,64,64)
        
        self.seg_output = Seg_Output(64,n_class)

    def forward(self, x):
        img = self.img_input(x[:, :3])
        img_1 = self.img_encoder_1(img)
        img_2 = self.img_encoder_2(img_1)
        img_3 = self.img_encoder_3(img_2)
 
        inf = self.inf_input(x[:,3:])
        inf_1a = self.inf_encoder_1(inf)
        inf_1a = self.inf_encoder_att_1(inf_1a)
        inf_2a = self.inf_encoder_2(inf_1a)
        inf_2a = self.inf_encoder_att_2(inf_2a)
        inf_3a = self.inf_encoder_3(inf_2a)
        inf_3a = self.inf_encoder_att_3(inf_3a)
        
        seg_3f = torch.cat([img_3, inf_3a], dim=1)
        seg_3f = self.conv(seg_3f)
        seg_3f = self.pam_att3(seg_3f, inf_3a)
        seg_3f = self.cam_att3(seg_3f)
        seg_2f = self.decode_block3(seg_3f)
        
        seg_3f = self.resconcate3(seg_3f,img_3,inf_3a)
        seg_3f = self.pam_att3(seg_3f,inf_3a)
        seg_3f = self.cam_att3(seg_3f)
        seg_2f = self.decode_block3(seg_3f)

        seg_2f = self.resconcate2(seg_2f,img_2,inf_2a)
        seg_2f = self.pam_att2(seg_2f,inf_2a)
        seg_2f = self.cam_att2(seg_2f)
        seg_1f = self.decode_block2(seg_2f)

        seg_1f = self.resconcate1(seg_1f,img_1,inf_1a)
        seg = self.decode_block1(seg_1f)
        seg = self.pam_att(seg,seg)
        seg = self.cam_att(seg)
        seg = self.seg_output(seg)
        return seg
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    import os
    device = 'cuda:3'
    model = INFAttNet(5).to(device)
    x = torch.randn((1, 6, 128, 128)).to(device)
    y0 = model(x)
    print(y0.shape)
    print(count_parameters(model))