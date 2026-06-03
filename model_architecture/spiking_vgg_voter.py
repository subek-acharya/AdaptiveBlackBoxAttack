"""
Spiking VGG for Voter Dataset
Wrapper around generic SpikingVGG with voter-specific defaults.
Supports grayscale images with flexible dimensions (40×50).
"""

import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import functional, neuron, layer

__all__ = [
    "SpikingVGGVoter",
    "spiking_vgg11_voter",
    "spiking_vgg11_bn_voter",
    "spiking_vgg13_voter",
    "spiking_vgg13_bn_voter",
    "spiking_vgg16_voter",
    "spiking_vgg16_bn_voter",
    "spiking_vgg19_voter",
    "spiking_vgg19_bn_voter",
]


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class SpikingVGGVoter(nn.Module):
    """
    Spiking VGG for Voter Dataset with flexible dimensions and grayscale support.
    
    Args:
        vgg_name: 'VGG11', 'VGG13', 'VGG16', or 'VGG19'
        imgH: Image height
        imgW: Image width
        num_classes: Number of output classes
        batch_norm: Whether to use batch normalization
        spiking_neuron: Spiking neuron type (e.g., neuron.IFNode)
        **kwargs: Additional arguments for spiking neuron
    """
    
    def __init__(
        self,
        vgg_name,
        imgH,
        imgW,
        num_classes,
        batch_norm=False,
        norm_layer=None,
        spiking_neuron: callable = None,
        init_weights=True,
        **kwargs,
    ):
        super(SpikingVGGVoter, self).__init__()
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        # Build feature layers
        self.features = self._make_layers(
            cfg[vgg_name],
            batch_norm=batch_norm,
            norm_layer=norm_layer,
            spiking_neuron=spiking_neuron,
            **kwargs,
        )
        
        # Average pool - Adaptive to ensure [N, C, 1, 1] output
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        
        # Calculate flatten size AFTER avgpool
        with torch.no_grad():
            x = torch.zeros(1, 1, imgH, imgW)  # Grayscale input
            out = self.features(x)
            out = self.avgpool(out)  # Apply avgpool to get correct shape
            flatten_size = out.view(1, -1).shape[1]  # Should be num_channels (512)
        
        # Classifier
        self.classifier = layer.Linear(flatten_size, num_classes)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """
        Forward pass supporting both single-step and multi-step modes.
        
        Single-step: input [N, C, H, W] → output [N, num_classes]
        Multi-step: input [T, N, C, H, W] → output [T, N, num_classes]
        """
        x = self.features(x)
        x = self.avgpool(x)
        
        # Handle both single-step and multi-step modes
        if len(x.shape) == 4:  # Single-step: [N, C, H, W]
            x = x.view(x.size(0), -1)  # [N, C]
        elif len(x.shape) == 5:  # Multi-step: [T, N, C, H, W]
            x = x.view(x.shape[0], x.shape[1], -1)  # [T, N, C]
        
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Initialize weights using Kaiming normal for Conv and normal for Linear."""
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layers(
        cfg_list,
        batch_norm=False,
        norm_layer=None,
        spiking_neuron: callable = None,
        **kwargs,
    ):
        """Build feature extraction layers."""
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        layers = []
        in_channels = 1  # Grayscale
        
        for v in cfg_list:
            if v == 'M':
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = layer.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [
                        conv2d,
                        norm_layer(v),
                        spiking_neuron(**deepcopy(kwargs))
                    ]
                else:
                    layers += [
                        conv2d,
                        spiking_neuron(**deepcopy(kwargs))
                    ]
                in_channels = v
        
        return nn.Sequential(*layers)


# --------------- Factory Functions ---------------

def _spiking_vgg_voter(
    vgg_name,
    imgH,
    imgW,
    num_classes,
    batch_norm=False,
    spiking_neuron: callable = None,
    **kwargs,
):
    """Internal factory function."""
    model = SpikingVGGVoter(
        vgg_name=vgg_name,
        imgH=imgH,
        imgW=imgW,
        num_classes=num_classes,
        batch_norm=batch_norm,
        spiking_neuron=spiking_neuron,
        **kwargs,
    )
    return model


def spiking_vgg11_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG11 without batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG11", imgH, imgW, num_classes, False, spiking_neuron, **kwargs)


def spiking_vgg11_bn_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG11 with batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG11", imgH, imgW, num_classes, True, spiking_neuron, **kwargs)


def spiking_vgg13_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG13 without batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG13", imgH, imgW, num_classes, False, spiking_neuron, **kwargs)


def spiking_vgg13_bn_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG13 with batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG13", imgH, imgW, num_classes, True, spiking_neuron, **kwargs)


def spiking_vgg16_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG16 without batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG16", imgH, imgW, num_classes, False, spiking_neuron, **kwargs)


def spiking_vgg16_bn_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG16 with batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG16", imgH, imgW, num_classes, True, spiking_neuron, **kwargs)


def spiking_vgg19_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG19 without batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG19", imgH, imgW, num_classes, False, spiking_neuron, **kwargs)


def spiking_vgg19_bn_voter(imgH, imgW, num_classes=2, spiking_neuron: callable = None, **kwargs):
    """Spiking VGG19 with batch normalization for Voter dataset."""
    return _spiking_vgg_voter("VGG19", imgH, imgW, num_classes, True, spiking_neuron, **kwargs)