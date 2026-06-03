"""
Spiking ResNet V2 for Voter Dataset
Adapted from the ResNet V2 architecture used in the voter classification task.
Supports grayscale images with flexible dimensions (40×50).

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from copy import deepcopy
from spikingjelly.activation_based import functional, neuron, layer

__all__ = [
    "SpikingResNetVoter",
    "spiking_resnet20_voter",
    "spiking_resnet56_voter",
    "spiking_resnet164_voter",
    "spiking_resnet1001_voter",
]


def _weights_init(m):
    """Initialize weights using Kaiming normal."""
    if isinstance(m, layer.Linear) or isinstance(m, layer.Conv2d):
        init.kaiming_normal_(m.weight)


class SpikingBasicBlockVoter(nn.Module):
    """
    Spiking Basic Block for ResNet V2 (Pre-activation) for Voter dataset.
    
    This follows the Keras ResNet V2 architecture:
    BN -> SN (instead of ReLU) -> Conv pattern
    """
    expansion = 1

    def __init__(
        self,
        res_block,
        activation,
        batch_normalization,
        in_planes,
        planes,
        stride,
        norm_layer=None,
        spiking_neuron: callable = None,
        **kwargs,
    ):
        super(SpikingBasicBlockVoter, self).__init__()
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        self.res_block = res_block
        self.activation = activation
        self.batch_normalization = batch_normalization

        # ResNet V2 architecture (pre-activation)
        if res_block == 0:
            self.bn1 = norm_layer(in_planes)
            self.conv1 = layer.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, bias=True)
        else:
            self.bn1 = norm_layer(planes)
            self.conv1 = layer.Conv2d(planes, in_planes, kernel_size=1, stride=stride, bias=True)

        # Spiking neuron for first activation
        self.sn1 = spiking_neuron(**deepcopy(kwargs)) if activation else None

        self.bn2 = norm_layer(in_planes)
        self.conv2 = layer.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))

        self.bn3 = norm_layer(in_planes)
        self.conv3 = layer.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=True)
        self.sn3 = spiking_neuron(**deepcopy(kwargs))

        # Skip connection
        self.shortcut = nn.Sequential()
        if res_block == 0:
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            )
        
        # Final activation after residual addition
        self.sn_out = spiking_neuron(**deepcopy(kwargs))

    def forward(self, x):
        # Pre-activation pattern: BN -> SN -> Conv
        if self.activation and self.batch_normalization:
            out = self.conv1(self.sn1(self.bn1(x)))
        elif self.activation and not self.batch_normalization:
            out = self.conv1(self.sn1(x))
        elif not self.activation and self.batch_normalization:
            out = self.conv1(self.bn1(x))
        else:
            out = self.conv1(x)
        
        out = self.conv2(self.sn2(self.bn2(out)))
        out = self.conv3(self.sn3(self.bn3(out)))
        out += self.shortcut(x)
        out = self.sn_out(out)
        
        return out


class SpikingResNetVoter(nn.Module):
    """
    Spiking ResNet V2 for Voter Dataset with flexible dimensions and grayscale support.
    
    Args:
        block: Block type (SpikingBasicBlockVoter)
        num_blocks: List of number of blocks per stage [stage1, stage2, stage3]
        imgH: Image height
        imgW: Image width
        num_classes: Number of output classes
        norm_layer: Normalization layer type
        spiking_neuron: Spiking neuron type (e.g., neuron.IFNode)
        **kwargs: Additional arguments for spiking neuron
    """
    
    def __init__(
        self,
        block,
        num_blocks,
        imgH,
        imgW,
        num_classes=2,
        norm_layer=None,
        spiking_neuron: callable = None,
        init_weights=True,
        **kwargs,
    ):
        super(SpikingResNetVoter, self).__init__()
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        self._norm_layer = norm_layer
        self.num_classes = num_classes
        self.in_planes = 16
        
        # Initial convolution (grayscale input)
        self.conv1 = layer.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = norm_layer(16)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        
        # Three stages (stacks) following ResNet V2 architecture
        # Stage 1
        in_planes = 16
        self.layer1 = self._make_layer(
            block, 0, in_planes, num_blocks[0], 
            spiking_neuron=spiking_neuron, **kwargs
        )
        
        # Stage 2
        in_planes = 64
        self.layer2 = self._make_layer(
            block, 1, in_planes, num_blocks[1],
            spiking_neuron=spiking_neuron, **kwargs
        )
        
        # Stage 3
        in_planes = 128
        self.layer3 = self._make_layer(
            block, 2, in_planes, num_blocks[2],
            spiking_neuron=spiking_neuron, **kwargs
        )
        
        # Final batch norm
        classifier_input_size = in_planes * 2  # 256
        self.bn2 = norm_layer(classifier_input_size)
        
        # Adaptive average pooling to handle flexible input sizes
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        
        # Calculate flatten size
        with torch.no_grad():
            x = torch.zeros(1, 1, imgH, imgW)
            out = self.sn1(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.bn2(out)
            out = self.avgpool(out)
            flatten_size = out.view(1, -1).shape[1]
        
        # Classifier
        self.fc = layer.Linear(flatten_size, num_classes)
        
        if init_weights:
            self.apply(_weights_init)

    def _make_layer(
        self,
        block,
        stage_num,
        in_planes,
        num_blocks,
        spiking_neuron: callable = None,
        **kwargs,
    ):
        """Build a stage (stack) of residual blocks."""
        norm_layer = self._norm_layer
        layers = []
        
        for res_block in range(num_blocks):
            # Setup following Keras ResNet V2 pattern
            activation = True
            batch_normalization = True
            strides = 1
            
            if stage_num == 0:
                planes = in_planes * 4  # 64
                if res_block == 0:  # First layer and first stage
                    activation = False
                    batch_normalization = False
            else:
                planes = in_planes * 2  # 128 or 256
                if res_block == 0:  # First layer but not first stage
                    strides = 2  # Downsample
            
            layers.append(
                block(
                    res_block,
                    activation,
                    batch_normalization,
                    in_planes,
                    planes,
                    strides,
                    norm_layer=norm_layer,
                    spiking_neuron=spiking_neuron,
                    **kwargs,
                )
            )
        
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass supporting both single-step and multi-step modes.
        
        Single-step: input [N, C, H, W] → output [N, num_classes]
        Multi-step: input [T, N, C, H, W] → output [T, N, num_classes]
        """
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn2(out)
        out = self.avgpool(out)
        
        # Handle both single-step and multi-step modes
        if len(out.shape) == 4:  # Single-step: [N, C, H, W]
            out = out.view(out.size(0), -1)  # [N, C]
        elif len(out.shape) == 5:  # Multi-step: [T, N, C, H, W]
            out = out.view(out.shape[0], out.shape[1], -1)  # [T, N, C]
        
        out = self.fc(out)
        return out


# --------------- Factory Functions ---------------

def _spiking_resnet_voter(
    num_blocks,
    imgH,
    imgW,
    num_classes,
    spiking_neuron: callable = None,
    **kwargs,
):
    """Internal factory function."""
    model = SpikingResNetVoter(
        block=SpikingBasicBlockVoter,
        num_blocks=num_blocks,
        imgH=imgH,
        imgW=imgW,
        num_classes=num_classes,
        spiking_neuron=spiking_neuron,
        **kwargs,
    )
    return model


def spiking_resnet20_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking ResNet-20 V2 for Voter dataset."""
    return _spiking_resnet_voter([2, 2, 2], imgH, imgW, num_classes, spiking_neuron, **kwargs)


def spiking_resnet56_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking ResNet-56 V2 for Voter dataset."""
    return _spiking_resnet_voter([6, 6, 6], imgH, imgW, num_classes, spiking_neuron, **kwargs)


def spiking_resnet164_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking ResNet-164 V2 for Voter dataset."""
    return _spiking_resnet_voter([18, 18, 18], imgH, imgW, num_classes, spiking_neuron, **kwargs)


def spiking_resnet1001_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking ResNet-1001 V2 for Voter dataset."""
    return _spiking_resnet_voter([111, 111, 111], imgH, imgW, num_classes, spiking_neuron, **kwargs)