# Adaptive Black-Box Adversarial Attack

Adaptive black-box adversarial attack framework for evaluating robustness of ballot classification systems. The attack trains a synthetic substitute model using oracle-labeled data, then executes six white-box attacks on the synthetic model to generate transferable adversarial examples.

## Overview

In this attack framework, the adversary has access to part of the training data and query access to the target model (oracle), receiving only hard-label predictions. The adversary queries the oracle to label training data, trains an independent synthetic classifier (VGG11), and then executes white-box attacks on the synthetic model — leveraging full parameter access — to craft adversarial examples that transfer to the oracle model.

This approach is based on the substitute model methodology introduced by Papernot et al. [1], extended with iterative data augmentation and six diverse white-box attacks spanning L∞, L2, L1, and L0 perturbation norms.

## Threat Model

| Property | Description |
|----------|-------------|
| Adversary Knowledge | No access to oracle architecture, weights, or gradients |
| Query Access | Hard-label predictions only (predicted class) |
| Training Data | Partial access to original training data |
| Goal | Generate adversarial examples that fool the oracle |

## Attack Pipeline


## Attacks Implemented

| Attack | Norm |
|--------|------|
| APGD-L∞ (ATA-L∞) | L∞ |
| APGD-L2 (ATA-L2) | L2 |
| APGD-L1 (ATA-L1) | L1 |
| L0-PGD (ATA-L0) | L0 |
| L0+Lσ-PGD (ATA-L0+Lσ) | L0 |
| L0+L∞-PGD (ATA-L0+L∞) | L0+L∞ |


## Oracle

| Model | Architecture | Type |
|-------|--------------|------|
| ResNet20-Combined | ResNet-20 | CNN |
| VGG16-Combined | VGG-16 | CNN |
| CaiT-Combined | CaiT | Transformer |
| SVM-Combined | Multi-Output SVM | Classical ML |
| SNN-VGG16-Combined | Spiking VGG-16 | SNN |
| SNN-ResNet20-Combined | Spiking ResNet-20 | SNN |
| xAI-VGG16-Combined | ProtoPNet (VGG-16) | Explainable AI |
| xAI-ResNet20-Combined | ProtoPNet (ResNet-20) | Explainable AI |
| MambaVision-L2-Combined | MambaVision-L2 | SSM + Transformer |

## Project Structure

```bash
AdaptiveBlackBoxAttack/
│
├── main.py                            # Main entry point for adaptive attack
├── AttackWrappersAdaptiveBlackBox.py   # Core adaptive attack logic
├── AttackFactory.py                    # Attack implementations (6 attacks)
├── AttackRunner.py                     # Runs all attacks and evaluates results
├── ModelFactory.py                     # Model loading factory for all architectures
├── DataLoaderGiant.py                  # Memory-efficient multi-dataloader management
├── config.py                           # Attack hyperparameters
├── constants.py                        # Paths and experiment configurations
├── utils.py                            # Shared utilities and data loaders
│
├── attacks/                            # Attack implementations
│   ├── linf_attack/
│   │   ├── APGD_Linf.py               # APGD-L∞ attack
│   │   └── FGSM.py                    # FGSM (for data augmentation)
│   ├── l1_attack/
│   │   ├── autoattack.py              # AutoAttack L1
│   │   └── autopgd_base.py            # APGD-L1 base
│   ├── l2_attack/
│   │   └── autopgd_base.py            # APGD-L2 base
│   └── l0_attack/
│       ├── L0_PGD.py                  # L0-PGD attack
│       ├── L0_Linf_PGD.py            # L0+L∞-PGD attack
│       └── L0_Sigma_PGD.py           # L0+Lσ-PGD attack
│
├── model_architecture/                 # Model architectures
│   ├── ResNet.py
│   ├── VGG.py
│   ├── cait.py
│   ├── MultiOutputSVM.py
│   ├── UNet.py
│   ├── spiking_vgg_voter.py
│   ├── spiking_resnet_voter.py
│   └── CarliniNetwork.py
│
├── checkpoint/                         # Pre-trained model checkpoints
├── data/                               # Datasets
└── README.md
```

## Usage

### Running Model Only Mode
```bash
# Edit main.py to set:
# ATTACK_MODE = "model_only"
# EXPERIMENTS_MODEL_ONLY = EXPERIMENTS_ALL

python main.py
```

### Running UNet+Model Mode
```bash
# Edit main.py to set:
# ATTACK_MODE = "unet"
# EXPERIMENTS_UNET_MODE = EXPERIMENTS_UNET_ALL

python main.py
```

### Running a Single Oracle Model
```bash
from main import run_attack_on_oracle
from constants import EXPERIMENTS_ALL

# Attack ResNet20
config = EXPERIMENTS_ALL["resnet20_combined"]
run_attack_on_oracle("resnet20_combined", config, "vgg11")
```

### Configuration
Attack hyperparameters are defined in *config.py*:
```python
# L∞ perturbation budgets
APGD_LINF_PARAMS = {
    "eps_1": 1/255,
    "eps_2": 2/255,
    "eps_4": 4/255,
}

# L0 sparsity budgets
L0_PGD_PARAMS = {
    "k_10": 10,
    "k_20": 20,
    "k_50": 50,
}
```

## References
- [1] Papernot, N., McDaniel, P. D., Goodfellow, I. J., Jha, S., Celik, Z. B., & Swami, A. (2017). [Practical Black-Box Attacks against Machine Learning](https://arxiv.org/abs/1602.02697). In *ACM AsiaCCS 2017*, pages 506–519.

- [2] Croce, F., & Hein, M. (2020). [Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks](https://arxiv.org/abs/2003.01690). In *International Conference on Machine Learning (ICML)*, PMLR.

- [3] Croce, F., & Hein, M. (2021). [Mind the Box: L1-APGD for Sparse Adversarial Attacks on Image Classifiers](https://arxiv.org/abs/2103.01208). In *International Conference on Machine Learning (ICML)*, pages 2201–2211, PMLR.

- [4] Croce, F., & Hein, M. (2019). [Sparse and Imperceivable Adversarial Attacks](https://arxiv.org/abs/1909.05040). In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 4724–4732.

- [5] Mahmood, K., Nguyen, P. H., Nguyen, L. M., Nguyen, T., & Van Dijk, M. (2022). [Besting the Black-Box: Barrier Zones for Adversarial Example Defense](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9663375). *IEEE Access*, 10, 1451–1474.