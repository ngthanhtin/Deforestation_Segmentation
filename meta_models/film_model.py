from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.nn import init
from pathlib import Path
import torchvision.models


#Modified from torchvision.models.resnet.Resnet
class ResNet(nn.Module):
    """ResNet model based on
     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385.pdf>`_
    
    Args:
        block (nn.Module): Basic block of the network (such as conv + bn + non-nonlinearity)
        layers (int list): Number of blocks to stack (network depth) after each downsampling step.
        num_classes (int): Number of GT classes.
        zero_init_residual (bool): if True, zero-initialize the last BN in each residual branch,
        so that the residual branch starts with zeros, and each residual block behaves like an identity.
        groups (int): Number of parallel branches in the network, see `"paper" <https://arxiv.org/pdf/1611.05431.pdf>`_
        width_per_group (int): base width of the blocks
        replace_stride_with_dilation (tuple): each element in the tuple indicates if we replace the 2x2 stride with a dilated convolution
        norm_layer (layer): Custom normalization layer, if not provided BatchNorm2d is used

    """
    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, torchvision.models.resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, torchvision.models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                torchvision.models.resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        preds = F.softmax(x, dim=1)
        # Remove background class
        preds = preds[:, 1:]
        return preds

    def forward(self, x):
        return self._forward_impl(x)


class resnet18(ResNet):
    def __init__(self, **kwargs):
        super().__init__(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        dropout_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"article" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, dropout_rate=0.3, num_classes=2, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = torchvision.models.densenet._DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=dropout_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = torchvision.models.densenet._Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        preds = F.softmax(out, dim=1)
        # Remove background class
        preds = preds[:, 1:]
        return preds


class densenet121(DenseNet):
    def __init__(self, **kwargs):
        super().__init__(32, (6, 12, 24, 16), 64)


class FiLMgenerator(Module):
    """The FiLM generator processes the conditioning information
    and produces parameters that describe how the target network should alter its computation.

    Here, the FiLM generator is a multi-layer perceptron.

    Args:
        n_features (int): Number of input channels.
        n_channels (int): Number of output channels.
        n_hid (int): Number of hidden units in layer.

    Attributes:
        linear1 (Linear): Input linear layer.
        sig (Sigmoid): Sigmoid function.
        linear2 (Linear): Hidden linear layer.
        linear3 (Linear): Output linear layer.
    """

    def __init__(self, n_features, n_channels, n_hid=64):
        super(FiLMgenerator, self).__init__()
        self.linear1 = nn.Linear(n_features, n_hid)
        self.sig = nn.Sigmoid()
        self.linear2 = nn.Linear(n_hid, n_hid // 4)
        self.linear3 = nn.Linear(n_hid // 4, n_channels * 2)

    def forward(self, x, shared_weights=None):
        if shared_weights is not None:  # weight sharing
            self.linear1.weight = shared_weights[0]
            self.linear2.weight = shared_weights[1]

        x = self.linear1(x)
        x = self.sig(x)
        x = self.linear2(x)
        x = self.sig(x)
        x = self.linear3(x)

        out = self.sig(x)
        return out, [self.linear1.weight, self.linear2.weight]

class FiLMlayer(Module):
    """Applies Feature-wise Linear Modulation to the incoming data
    .. seealso::
        Perez, Ethan, et al. "Film: Visual reasoning with a general conditioning layer."
        Thirty-Second AAAI Conference on Artificial Intelligence. 2018.

    Args:
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        n_channels (int): Number of output channels.

    Attributes:
        batch_size (int): Batch size.
        height (int): Image height.
        width (int): Image width.
        feature_size (int): Number of features in data.
        generator (FiLMgenerator): FiLM network.
        gammas (float): Multiplicative term of the FiLM linear modulation.
        betas (float): Additive term of the FiLM linear modulation.
    """

    def __init__(self, n_metadata, n_channels):
        super(FiLMlayer, self).__init__()

        self.batch_size = None
        self.height = None
        self.width = None
        self.depth = None
        self.feature_size = None
        self.generator = FiLMgenerator(n_metadata, n_channels)
        # Add the parameters gammas and betas to access them out of the class.
        self.gammas = None
        self.betas = None

    def forward(self, feature_maps, context, w_shared):
        data_shape = feature_maps.data.shape
        if len(data_shape) == 4:
            _, self.feature_size, self.height, self.width = data_shape
        elif len(data_shape) == 5:
            _, self.feature_size, self.height, self.width, self.depth = data_shape
        else:
            raise ValueError("Data should be either 2D (tensor length: 4) or 3D (tensor length: 5), found shape: {}".format(data_shape))

        # if torch.cuda.is_available():
        #     context = torch.Tensor(context).cuda()
        # else:
        context = torch.Tensor(context)

        # Estimate the FiLM parameters using a FiLM generator from the contioning metadata
        film_params, new_w_shared = self.generator(context, w_shared)

        # FiLM applies a different affine transformation to each channel,
        # consistent accross spatial locations
        if len(data_shape) == 4:
            film_params = film_params.unsqueeze(-1).unsqueeze(-1)
            film_params = film_params.repeat(1, 1, self.height, self.width)
        else:
            film_params = film_params.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            film_params = film_params.repeat(1, 1, self.height, self.width, self.depth)

        self.gammas = film_params[:, :self.feature_size, ]
        self.betas = film_params[:, self.feature_size:, ]

        # Apply the linear modulation
        output = self.gammas * feature_maps + self.betas

        return output, new_w_shared
    
class DownConv(Module):
    """Two successive series of down convolution, batch normalization and dropout in 2D.
    Used in U-Net's encoder.

    Args:
        in_feat (int): Number of channels in the input image.
        out_feat (int): Number of channels in the output image.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.

    Attributes:
        conv1 (Conv2d): First 2D down convolution with kernel size 3 and padding of 1.
        conv1_bn (BatchNorm2d): First 2D batch normalization.
        conv1_drop (Dropout2d): First 2D dropout.
        conv2 (Conv2d): Second 2D down convolution with kernel size 3 and padding of 1.
        conv2_bn (BatchNorm2d): Second 2D batch normalization.
        conv2_drop (Dropout2d): Second 2D dropout.
    """

    def __init__(self, in_feat, out_feat, dropout_rate=0.3, bn_momentum=0.1, is_2d=True):
        super(DownConv, self).__init__()
        if is_2d:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
            dropout = nn.Dropout2d
        else:
            conv = nn.Conv3d
            bn = nn.InstanceNorm3d
            dropout = nn.Dropout3d

        self.conv1 = conv(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = bn(out_feat, momentum=bn_momentum)
        self.conv1_drop = dropout(dropout_rate)

        self.conv2 = conv(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = bn(out_feat, momentum=bn_momentum)
        self.conv2_drop = dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        return x


class UpConv(Module):
    """2D down convolution.
    Used in U-Net's decoder.

    Args:
        in_feat (int): Number of channels in the input image.
        out_feat (int): Number of channels in the output image.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.

    Attributes:
        downconv (DownConv): Down convolution.
    """

    def __init__(self, in_feat, out_feat, dropout_rate=0.3, bn_momentum=0.1, is_2d=True):
        super(UpConv, self).__init__()
        self.is_2d = is_2d
        self.downconv = DownConv(in_feat, out_feat, dropout_rate, bn_momentum, is_2d)

    def forward(self, x, y):
        # For retrocompatibility purposes
        if not hasattr(self, "is_2d"):
            self.is_2d = True
        mode = 'bilinear' if self.is_2d else 'trilinear'
        dims = -2 if self.is_2d else -3
        x = F.interpolate(x, size=y.size()[dims:], mode=mode, align_corners=True)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x


class Encoder(Module):
    """Encoding part of the U-Net model.
    It returns the features map for the skip connections

    Args:
        in_channel (int): Number of channels in the input image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.
        n_filters (int):  Number of base filters in the U-Net.

    Attributes:
        depth (int): Number of down convolutions minus bottom down convolution.
        down_path (ModuleList): List of module operations done during encoding.
        conv_bottom (DownConv): Bottom down convolution.
        film_bottom (FiLMlayer): FiLM layer applied to bottom convolution.
    """

    def __init__(self, in_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1, n_metadata=None, film_layers=None,
                 is_2d=True, n_filters=64):
        super(Encoder, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        # first block
        self.down_path.append(DownConv(in_channel, n_filters, dropout_rate, bn_momentum, is_2d))
        self.down_path.append(FiLMlayer(n_metadata, n_filters) if film_layers and film_layers[0] else None)
        max_pool = nn.MaxPool2d if is_2d else nn.MaxPool3d
        self.down_path.append(max_pool(2))

        for i in range(depth - 1):
            self.down_path.append(DownConv(n_filters, n_filters * 2, dropout_rate, bn_momentum, is_2d))
            self.down_path.append(FiLMlayer(n_metadata, n_filters * 2) if film_layers and film_layers[i + 1] else None)
            self.down_path.append(max_pool(2))
            n_filters = n_filters * 2

        # Bottom
        self.conv_bottom = DownConv(n_filters, n_filters, dropout_rate, bn_momentum, is_2d)
        self.film_bottom = FiLMlayer(n_metadata, n_filters) if film_layers and film_layers[self.depth] else None

    def forward(self, x, context=None):
        features = []

        # First block
        x = self.down_path[0](x)
        if self.down_path[1]:
            x, w_film = self.down_path[1](x, context, None)
        features.append(x)
        x = self.down_path[2](x)

        # Down-sampling path (other blocks)
        for i in range(1, self.depth):
            x = self.down_path[i * 3](x)
            if self.down_path[i * 3 + 1]:
                x, w_film = self.down_path[i * 3 + 1](x, context, None if 'w_film' not in locals() else w_film)
            features.append(x)
            x = self.down_path[i * 3 + 2](x)

        # Bottom level
        x = self.conv_bottom(x)
        if self.film_bottom:
            x, w_film = self.film_bottom(x, context, None if 'w_film' not in locals() else w_film)
        features.append(x)

        return features, None if 'w_film' not in locals() else w_film


class Decoder(Module):
    """Decoding part of the U-Net model.

    Args:
        out_channel (int): Number of channels in the output image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.
        hemis (bool): Boolean indicating if HeMIS is on or not.
        final_activation (str): Choice of final activation between "sigmoid", "relu" and "softmax".
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.
        n_filters (int):  Number of base filters in the U-Net.

    Attributes:
        depth (int): Number of down convolutions minus bottom down convolution.
        out_channel (int): Number of channels in the output image.
        up_path (ModuleList): List of module operations done during decoding.
        last_conv (Conv2d): Last convolution.
        last_film (FiLMlayer): FiLM layer applied to last convolution.
        softmax (Softmax): Softmax layer that can be applied as last layer.
    """

    def __init__(self, out_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1,
                 n_metadata=None, film_layers=None, hemis=False, final_activation="sigmoid", is_2d=True, n_filters=64):
        super(Decoder, self).__init__()
        self.depth = depth
        self.out_channel = out_channel
        self.final_activation = final_activation
        # Up-Sampling path
        self.up_path = nn.ModuleList()
        if hemis:
            in_channel = n_filters * 2 ** self.depth
            self.up_path.append(UpConv(in_channel * 2, n_filters * 2 ** (self.depth - 1), dropout_rate, bn_momentum,
                                       is_2d))
            if film_layers and film_layers[self.depth + 1]:
                self.up_path.append(FiLMlayer(n_metadata, n_filters * 2 ** (self.depth - 1)))
            else:
                self.up_path.append(None)
            # self.depth += 1
        else:
            in_channel = n_filters * 2 ** self.depth

            self.up_path.append(UpConv(in_channel, n_filters * 2 ** (self.depth - 1), dropout_rate, bn_momentum, is_2d))
            if film_layers and film_layers[self.depth + 1]:
                self.up_path.append(FiLMlayer(n_metadata, n_filters * 2 ** (self.depth - 1)))
            else:
                self.up_path.append(None)

        for i in range(1, depth):
            in_channel //= 2

            self.up_path.append(
                UpConv(in_channel + n_filters * 2 ** (self.depth - i - 1 + int(hemis)), n_filters * 2 ** (self.depth - i - 1),
                       dropout_rate, bn_momentum, is_2d))
            if film_layers and film_layers[self.depth + i + 1]:
                self.up_path.append(FiLMlayer(n_metadata, n_filters * 2 ** (self.depth - i - 1)))
            else:
                self.up_path.append(None)

        # Last Convolution
        conv = nn.Conv2d if is_2d else nn.Conv3d
        self.last_conv = conv(in_channel // 2, out_channel, kernel_size=3, padding=1)
        self.last_film = FiLMlayer(n_metadata, self.out_channel) if film_layers and film_layers[-1] else None
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, context=None, w_film=None):
        x = features[-1]

        for i in reversed(range(self.depth)):
            x = self.up_path[-(i + 1) * 2](x, features[i])
            if self.up_path[-(i + 1) * 2 + 1]:
                x, w_film = self.up_path[-(i + 1) * 2 + 1](x, context, w_film)

        # Last convolution
        x = self.last_conv(x)
        if self.last_film:
            x, w_film = self.last_film(x, context, w_film)

        if hasattr(self, "final_activation") and self.final_activation not in ["softmax", "relu", "sigmoid"]:
            raise ValueError("final_activation value has to be either softmax, relu, or sigmoid")
        elif hasattr(self, "final_activation") and self.final_activation == "softmax":
            preds = self.softmax(x)
        elif hasattr(self, "final_activation") and self.final_activation == "relu":
            preds = nn.ReLU()(x) / nn.ReLU()(x).max()
            # If nn.ReLU()(x).max()==0, then nn.ReLU()(x) will also ==0. So, here any zero division will always be 0/0.
            # For float values, 0/0=nan. So, we can handle zero division (without checking data!) by setting nans to 0.
            preds[torch.isnan(preds)] = 0.
            # If model multiclass
            if self.out_channel > 1:
                class_sum = preds.sum(dim=1).unsqueeze(1)
                # Avoid division by zero
                class_sum[class_sum == 0] = 1
                preds /= class_sum
        elif hasattr(self, "final_activation") and self.final_activation == "sigmoid":
            preds = torch.sigmoid(x)
        else:
            preds = x

        # If model multiclass
        # if self.out_channel > 1:
        #     # Remove background class
        #     preds = preds[:, 1:, ]

        return preds


class Unet(Module):
    """A reference U-Net model.

    .. seealso::
        Ronneberger, O., et al (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation
        ArXiv link: https://arxiv.org/abs/1505.04597

    Args:
        in_channel (int): Number of channels in the input image.
        out_channel (int): Number of channels in the output image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        final_activation (str): Choice of final activation between "sigmoid", "relu" and "softmax".
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.
        n_filters (int):  Number of base filters in the U-Net.
        **kwargs:

    Attributes:
        encoder (Encoder): U-Net encoder.
        decoder (Decoder): U-net decoder.
    """

    def __init__(self, in_channel=1, out_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1, final_activation='sigmoid',
                 is_2d=True, n_filters=64, **kwargs):
        super(Unet, self).__init__()

        # Encoder path
        self.encoder = Encoder(in_channel=in_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                               is_2d=is_2d, n_filters=n_filters)

        # Decoder path
        self.decoder = Decoder(out_channel=out_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                               final_activation=final_activation, is_2d=is_2d, n_filters=n_filters)

    def forward(self, x):
        features, _ = self.encoder(x)
        preds = self.decoder(features)

        return preds


class FiLMedUnet(Unet):
    """U-Net network containing FiLM layers to condition the model with another data type (i.e. not an image).

    Args:
        n_channel (int): Number of channels in the input image.
        out_channel (int): Number of channels in the output image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.
        is_2d (bool): Indicates dimensionality of model.
        n_filters (int):  Number of base filters in the U-Net.
        **kwargs:

    Attributes:
        encoder (Encoder): U-Net encoder.
        decoder (Decoder): U-net decoder.
    """

    def __init__(self, in_channel=1, out_channel=1, depth=3, dropout_rate=0.3,
                 bn_momentum=0.1, n_metadata=None, film_layers=None, is_2d=True, n_filters=64, **kwargs):
        super().__init__(in_channel=in_channel, out_channel=out_channel, depth=depth,
                         dropout_rate=dropout_rate, bn_momentum=bn_momentum)

        # Verify if the length of boolean FiLM layers corresponds to the depth
        if film_layers:
            if len(film_layers) != 2 * depth + 2:
                raise ValueError("The number of FiLM layers {} entered does not correspond to the "
                                 "UNet depth. There should 2 * depth + 2 layers.".format(len(film_layers)))
        else:
            film_layers = [0] * (2 * depth + 2)
        # Encoder path
        self.encoder = Encoder(in_channel=in_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                               n_metadata=n_metadata, film_layers=film_layers, is_2d=is_2d, n_filters=n_filters)
        # Decoder path
        self.decoder = Decoder(out_channel=out_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                               n_metadata=n_metadata, film_layers=film_layers, is_2d=is_2d, n_filters=n_filters)

    def forward(self, x, context=None):
        features, w_film = self.encoder(x, context)
        preds = self.decoder(features, context, w_film)

        return preds


def test_unet():
    x = torch.randn((3, 6, 320, 320))
    model = Unet(in_channel=6, out_channel=5)
    preds = model(x)
    print(preds.shape)

def test_filmunet():
    
    MANUFACTURER_CATEGORY = {'Siemens': 0, 'Philips': 1, 'GE': 2}
    CONTRAST_CATEGORY = {"T1w": 0, "T2w": 1, "T2star": 2,
                    "acq-MToff_MTS": 3, "acq-MTon_MTS": 4, "acq-T1w_MTS": 5}
    x = torch.randn((3, 6, 320, 320))
    context = torch.randn((3, 7))
    model = FiLMedUnet(in_channel=6, out_channel=5, film_layers=[1,0,1,0,0,1,1,1], n_metadata=7)
    preds = model(x,context)
    print(preds.shape)

test_filmunet()