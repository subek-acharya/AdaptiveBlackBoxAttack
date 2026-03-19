import torch
import torch.nn as nn
from typing import Union, List, Tuple, Optional
from model_architecture import ResNet, cait, VGG, MultiOutputSVM, CarliniNetwork

"""
ModelFactory - Factory class for creating neural network models

Supported Models:
-----------------
    Model Name      | Parameter String  | Checkpoint Path Format
    ----------------|-------------------|---------------------------
    ResNet20        | "resnet"          | str (single path)
    CaiT            | "cait"            | str (single path)
    VGG16           | "vgg16"           | str (single path)
    CarliniNetwork  | "carlini"         | str (single path)
    Multi-Output SVM| "svm"             | List[str] (two paths: [base_path, multi_path])

Usage Examples:
---------------
    # With checkpoint (trained model)
    model = ModelFactory().get_model("resnet", "./checkpoint/resnet_model.th")
    model = ModelFactory().get_model("cait", "./checkpoint/cait_model.th")
    model = ModelFactory().get_model("vgg", "./checkpoint/vgg_model.th")
    model = ModelFactory().get_model("carlini", "./checkpoint/carlini_model.th")
    model = ModelFactory().get_model("svm", ["./checkpoint/svm_base.th", "./checkpoint/svm_multi.th"])

    # Without checkpoint (untrained model for training)
    model = ModelFactory().get_model("resnet")
    model = ModelFactory().get_model("cait")
    model = ModelFactory().get_model("vgg")
    model = ModelFactory().get_model("carlini")
    model = ModelFactory().get_model("svm")

# --------------- Model PATHS for this repo------------------
# ResNet model
resnet_C_path = "./checkpoint/ModelResNet20-VotingCombined-v2-Grayscale-Run1.th"
# CaiT model
cait_C_path = "./checkpoint/ModelCaiT-trCombined-v2-valCombined-v2-Grayscale-Run1.th"
# VGG model
vgg_C_path = "./checkpoint/ModelVgg16-C2.th"
# SVM model
svm_C_base = "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/base_pytorch_svm_combined_v2.pth"
svm_C_multi = "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/multi_output_svm_combined_v2.pth"
"""

class ModelFactory:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def get_model(
        self,
        model_name: str,
        checkpoint_path: Optional[Union[str, List[str]]] = None,
    ) -> nn.Module:
        model_name = model_name.lower()

        if "resnet" in model_name:
            return self._create_resnet(checkpoint_path)
        elif "cait" in model_name:
            return self._create_cait(checkpoint_path)
        elif "vgg11" in model_name:
            return self._create_vgg11(checkpoint_path)
        elif "vgg16" in model_name:
            return self._create_vgg16(checkpoint_path)
        elif "carlini" in model_name:
            return self._create_carlini(checkpoint_path)
        elif "svm" in model_name:
            if checkpoint_path is None:
                return self._create_svm(None, None)
            elif isinstance(checkpoint_path, (list, tuple)) and len(checkpoint_path) == 2:
                return self._create_svm(checkpoint_path[0], checkpoint_path[1])
            else:
                raise ValueError(
                    "SVM requires a list/tuple of two paths: [base_path, multi_path]"
                )
        else:
            raise ValueError(f"Model '{model_name}' not recognized.")

    def _create_resnet(
        self,
        checkpoint_path: Optional[str] = None,
        input_size=[1, 1, 40, 50],
        num_classes=2,
        dropout=0.0,
    ) -> nn.Module:
        model = ResNet.resnet20(input_size, dropout, num_classes).to(self.device)
        
        if checkpoint_path is not None:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
        
        return model

    def _create_cait(
        self, 
        checkpoint_path: Optional[str] = None, 
        num_classes=2
    ) -> nn.Module:
        model = cait.CaiT(
            image_size=(40, 50),
            patch_size=5,
            num_classes=num_classes,
            num_channels=1,
            dim=512,
            depth=16,
            cls_depth=2,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05,
        ).to(self.device)

        if checkpoint_path is not None:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            model.load_state_dict(checkpoint["state_dict"])

            if hasattr(model, "patch_transformer"):
                model.patch_transformer.layer_dropout = 0.0
            if hasattr(model, "cls_transformer"):
                model.cls_transformer.layer_dropout = 0.0

            model.eval()
        
        return model

    def _create_vgg11(
        self, 
        checkpoint_path: Optional[str] = None, 
        num_classes=2
    ) -> nn.Module:
        
        model = VGG.VGG("VGG11", 40, 50, num_classes).to(self.device)
        if checkpoint_path is not None:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = raw.get("state_dict", raw)
            state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            model.eval()
        return model

    def _create_vgg16(
        self, 
        checkpoint_path: Optional[str] = None, 
        num_classes=2
    ) -> nn.Module:
        
        model = VGG.VGG("VGG16", 40, 50, num_classes).to(self.device)
        if checkpoint_path is not None:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = raw.get("state_dict", raw)
            state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            model.eval()
        return model

    def _create_carlini(
        self,
        checkpoint_path: Optional[str] = None,
        img_h: int = 40,
        img_w: int = 50,
        num_channels: int = 1,
        num_classes: int = 2,
    ) -> nn.Module:
        model = CarliniNetwork.CarliniNetwork(
            imgH=img_h,
            imgW=img_w,
            numChannels=num_channels,
            numClasses=num_classes,
        ).to(self.device)

        if checkpoint_path is not None:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.eval()

        return model

    def _create_svm(
        self, 
        base_path: Optional[str] = None, 
        multi_path: Optional[str] = None
    ) -> nn.Module:
        input_dim = 1 * 40 * 50
        
        if base_path is not None and multi_path is not None:
            base_state = torch.load(base_path, map_location="cpu", weights_only=False)
            model = MultiOutputSVM.MultiOutputSVM(input_dim, base_state).to(self.device)
            multi_state = torch.load(multi_path, map_location="cpu", weights_only=False)
            model.load_state_dict(multi_state)
            model.eval()
        else:
            # Create untrained SVM (you may need to adjust this based on your SVM implementation)
            model = MultiOutputSVM.MultiOutputSVM(input_dim, None).to(self.device)
        
        return model