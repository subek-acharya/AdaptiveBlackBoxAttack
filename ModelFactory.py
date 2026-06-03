import torch
import torch.nn as nn

from typing import Union, List, Tuple, Optional
import sys
from pathlib import Path

from spikingjelly.activation_based import surrogate, neuron, functional

from model_architecture import ResNet, cait, VGG, MultiOutputSVM
from model_architecture.UNet import UNet
from model_architecture.spiking_vgg_voter import spiking_vgg16_bn_voter
from model_architecture.spiking_resnet_voter import spiking_resnet20_voter

# ----------- Wrapper Class ---------------------
class SNNWrapper(nn.Module):
    """
    Wrapper that handles time dimension internally.
    Allows using existing utils.py functions without modification.
    """
    
    def __init__(self, snn_model, T=4):
        super(SNNWrapper, self).__init__()
        self.snn = snn_model
        self.T = T
    
    def forward(self, x):
        # Add time dimension: [N, C, H, W] → [T, N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        
        # Forward through SNN
        out_seq = self.snn(x_seq)
        
        # Average over time
        out = out_seq.mean(0)
        
        # Reset membrane
        functional.reset_net(self.snn)
        
        return out

class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            if "logits" in output:
                return output["logits"]
            raise KeyError("Model output dict missing 'logits' key.")
        if isinstance(output, tuple):
            return output[0]
        return output

class UNetModelWrapper(nn.Module):
    """
    Wrapper that passes input through UNet (denoiser) then through classifier model.
    
    Flow: Input → UNet → Model → Output
    
    This allows attacking the combined UNet+Model as a single unit.
    """
    
    def __init__(self, unet: nn.Module, model: nn.Module):
        super().__init__()
        self.unet = unet
        self.model = model
    
    def forward(self, x):
        # Pass through UNet (denoiser)
        cleaned = self.unet(x)
        
        # Handle if UNet returns tuple/list
        if isinstance(cleaned, (tuple, list)):
            cleaned = cleaned[0]
        
        # Clamp to valid range
        cleaned = cleaned.clamp(0.0, 1.0)
        
        # Pass through classifier
        output = self.model(cleaned)
        
        return output

# ----------------- MODEL FACTORY -----------------
class ModelFactory:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Base directory for resolving paths
        self.base_dir = Path(__file__).resolve().parent

    def get_model(
        self,
        model_name: str,
        checkpoint_path: Union[str, List[str]] = None,
    ) -> nn.Module:
        model_name = model_name.lower()

        if "unet" in model_name and "+" not in model_name:
            # Pure UNet model
            return self._create_unet(checkpoint_path)
        elif "expv2" in model_name or "explainable" in model_name:
            return self._create_ppnet_v2_direct(
                checkpoint_path,
            )
        elif "snn_resnet" in model_name or "snn-resnet" in model_name:
            return self._create_snn_resnet(
                checkpoint_path,
            )
        elif "snn_vgg" in model_name or "snn-vgg" in model_name:
            return self._create_snn_vgg(
                checkpoint_path,
            )
        elif "carlini" in model_name:
            return self._create_carlini(
                checkpoint_path
            )
        elif "resnet" in model_name:
            return self._create_resnet(
                checkpoint_path,
            )
        elif "cait" in model_name:
            return self._create_cait(
                checkpoint_path,
            )
        elif "vgg11" in model_name:
            # VGG11 must come before vgg16 check
            return self._create_vgg11(
                checkpoint_path,
            )
        elif "vgg" in model_name:
            return self._create_vgg16(
                checkpoint_path,
            )
        elif "svm" in model_name:
            if isinstance(checkpoint_path, (list, tuple)) and len(checkpoint_path) == 2:
                return self._create_svm(checkpoint_path[0], checkpoint_path[1])
            else:
                raise ValueError(
                    "SVM requires a list/tuple of two paths: [base_path, multi_path]"
                )
        elif "mamba" in model_name or "mambavision" in model_name:
            return self._create_mambavision(
                checkpoint_path,
                )
        else:
            raise ValueError(f"Model '{model_name}' not recognized.")

    def get_unet_model_wrapper(
        self,
        model_name: str,
        model_checkpoint: Union[str, List[str]],
        unet_checkpoint: str,
    ) -> nn.Module:
        """
        Create a UNet+Model wrapper for defense evaluation.
        
        Args:
            model_name: Name of the classifier model
            model_checkpoint: Path to classifier checkpoint
            unet_checkpoint: Path to UNet checkpoint
            
        Returns:
            UNetModelWrapper that combines UNet denoiser with classifier
        """
        # Load UNet
        unet = self._create_unet(unet_checkpoint)
        
        # Load classifier model
        model = self.get_model(model_name, model_checkpoint)
        
        # Create wrapper
        wrapper = UNetModelWrapper(unet, model).to(self.device)
        wrapper.eval()
        
        return wrapper

    def _create_unet(self, checkpoint_path: str) -> nn.Module:
        """Load UNet autoencoder/denoiser model."""
        unet = UNet().to(self.device)
        
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        
        unet.load_state_dict(state)
        unet.eval()
        
        return unet


    def _create_resnet(
        self,
        checkpoint_path: str,
        input_size=[1, 1, 40, 50],
        num_classes=2,
        dropout=0.0,
    ) -> nn.Module:
        model = ResNet.resnet20(input_size, dropout, num_classes).to(self.device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def _create_cait(self, checkpoint_path: str, num_classes=2) -> nn.Module:
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
        """Create VGG11 model (used as synthetic model for transfer attacks)."""
        model = VGG.VGG("VGG11", 40, 50, num_classes).to(self.device)
        
        if checkpoint_path is not None:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = raw.get("state_dict", raw)
            state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            model.eval()
        
        return model

    def _create_vgg16(self, checkpoint_path: str, num_classes=2) -> nn.Module:
        model = VGG.VGG("VGG16", 40, 50, num_classes).to(self.device)

        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)

        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    def _create_svm(self, base_path: str, multi_path: str) -> nn.Module:
        input_dim = 1 * 40 * 50
        base_state = torch.load(base_path, map_location="cpu", weights_only=False)

        model = MultiOutputSVM.MultiOutputSVM(input_dim, base_state).to(self.device)

        multi_state = torch.load(multi_path, map_location="cpu", weights_only=False)
        model.load_state_dict(multi_state)
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

    def _create_snn_vgg(
        self,
        checkpoint_path: str,
        imgH: int = 40,
        imgW: int = 50,
        num_classes: int = 2,
        T: int = 4,
    ) -> nn.Module:
        """Create Spiking VGG16-BN model with SNNWrapper."""
        snn_model = spiking_vgg16_bn_voter(
            imgH=imgH,
            imgW=imgW,
            num_classes=num_classes,
            spiking_neuron=neuron.IFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )
        functional.set_step_mode(snn_model, 'm')
        
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        snn_model.load_state_dict(checkpoint['model'])
        
        model = SNNWrapper(snn_model, T=T)
        model = model.to(self.device)
        model.eval()
        return model

    def _create_snn_resnet(
        self,
        checkpoint_path: str,
        imgH: int = 40,
        imgW: int = 50,
        num_classes: int = 2,
        T: int = 4,
    ) -> nn.Module:
        """Create Spiking ResNet-20 model with SNNWrapper."""
        snn_model = spiking_resnet20_voter(
            imgH=imgH,
            imgW=imgW,
            num_classes=num_classes,
            spiking_neuron=neuron.IFNode,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )
        functional.set_step_mode(snn_model, 'm')
        
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        snn_model.load_state_dict(checkpoint['model'])
        
        model = SNNWrapper(snn_model, T=T)
        model = model.to(self.device)
        model.eval()
        return model

    def _create_ppnet_v2_direct(self, checkpoint_path: str) -> nn.Module:
        """
        Load Explainable AI (ProtoPNet v2) model.
        
        Directory structure:
        Thesis/
        ├── Linf-BlackBoxAttack/
        │   └── ModelFactory.py       ← We are here (self.base_dir)
        └── Explanaible_AI/
            ├── cosine-is-almost/
            │   └── protopnext/
            │       └── protopnet/
            └── models/
                └── architecture/
                    └── ResNet.py
        """
        # ═══════════════════════════════════════════════════════════════
        # Path to Explanaible_AI directory (sibling of Linf-BlackBoxAttack)
        # ═══════════════════════════════════════════════════════════════
        
        # self.base_dir = Linf-BlackBoxAttack/
        # self.base_dir.parent = Thesis/
        # _EXPLAINABLE_DIR = Thesis/Explanaible_AI/
        
        _EXPLAINABLE_DIR = self.base_dir.parent / "Explanaible_AI"
        
        # Verify the directory exists
        if not _EXPLAINABLE_DIR.exists():
            raise FileNotFoundError(
                f"Explanaible_AI directory not found at: {_EXPLAINABLE_DIR}\n"
                f"Expected structure: {self.base_dir.parent}/Explanaible_AI/"
            )
        
        # Path to cosine-is-almost repo
        _COSINE_DIR = _EXPLAINABLE_DIR / "cosine-is-almost"
        _PPNEXT_DIR = _COSINE_DIR / "protopnext"
        
        # Path to base directory (for models.architecture.ResNet)
        _BASE_DIR = _EXPLAINABLE_DIR

        # ═══════════════════════════════════════════════════════════════
        # Add ALL required paths to sys.path
        # ═══════════════════════════════════════════════════════════════
        
        paths_to_add = [
            str(_PPNEXT_DIR),    # For: from protopnet.* import ...
            str(_COSINE_DIR),    # For: other cosine-is-almost imports
            str(_BASE_DIR),      # For: from models.architecture.ResNet import ...
        ]
        
        paths_added = []
        for p in paths_to_add:
            if p not in sys.path:
                sys.path.insert(0, p)
                paths_added.append(p)

        try:
            ppnet = torch.load(
                str(checkpoint_path), map_location=self.device, weights_only=False
            )
            ppnet = ppnet.to(self.device)
            ppnet.eval()
            wrapped = LogitsOnlyWrapper(ppnet).to(self.device)
            wrapped.eval()
            return wrapped

        except Exception as e:
            print(f"Failed to load v2 PPNet from {checkpoint_path}: {e}")
            raise
        finally:
            # Clean up added paths
            for p in paths_added:
                if p in sys.path:
                    sys.path.remove(p)
                    
    def _create_mambavision(
        self,
        checkpoint_path: Optional[str] = None,
        model_variant: str = "mamba_vision_L2",
        num_classes: int = 2,
    ) -> nn.Module:
        """Load MambaVision model with grayscale adaptation."""
        from mambavision import create_model as create_mamba_model

        # Create architecture
        model = create_mamba_model(model_variant, pretrained=False, num_classes=num_classes)

        # Adapt first conv for grayscale (3 → 1 channel)
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                module.in_channels = 1
                module.weight = nn.Parameter(module.weight[:, :1, :, :].clone())
                break

        # Load checkpoint
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present
            state_dict = {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }

            model.load_state_dict(state_dict, strict=False)

        model = model.to(self.device)
        model.eval()
        return model
