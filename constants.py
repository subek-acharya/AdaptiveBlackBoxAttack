"""
constants.py

Configuration file containing paths and settings for model experiments.

This file defines:
    - CHECKPOINTS: Paths to pre-trained model checkpoint files
    - VAL_DATASETS: Paths to validation dataset files
    - SCANNED_DATASETS: Paths to scanned bubble datasets (for UNet experiments)
    - EXPERIMENTS: Model-dataset configurations for running experiments
    - EXPERIMENTS_UNET_*: UNet + Model configurations using scanned data
"""

# -------------------------- Checkpoints for all models -----------------------------------------
CHECKPOINTS = {
    "resnet20_combined": "./checkpoint/ModelResNet20-VotingCombined-v2-Grayscale-Run1.th",
    "vgg16_combined": "./checkpoint/ModelVgg16-C2.th",
    "cait_combined": "./checkpoint/ModelCaiT-trCombined-v2-valCombined-v2-Grayscale-Run1.th",
    "svm_combined": [
        "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/base_pytorch_svm_combined_v2.pth",
        "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/multi_output_svm_combined_v2.pth",
    ],
    "snn_vgg16_combined": "./checkpoint/spiking_vgg16_bn_voter.pth",
    "snn_resnet20_combined": "./checkpoint/spiking_resnet20_voter.pth",
    "expv2_resnet20": "./checkpoint/Explainable_ResNet20.pth",
    "expv2_vgg16": "./checkpoint/Explainable_VGG16.pth",
    "mambavision_combined": "./checkpoint/mamba_model.pth",
    "unet": "./checkpoint/UNet.th",
}

# ------------------------ UNet Checkpoint ----------------------------------
UNET_CHECKPOINT = CHECKPOINTS["unet"]

# ------------------------ Training and Validation datasets ---------------------------------------
VAL_DATASETS = {
    "OnlyBubbles Val": "./data/kaleel_final_dataset_val_OnlyBubbles_Grayscale.pth",
    "Combined Val": "./data/kaleel_final_dataset_val_Combined_Grayscale.pth",
}

TRAIN_DATASETS = {
    "OnlyBubbles Train": "./data/kaleel_final_dataset_train_OnlyBubbles_Grayscale.pth",
    "Combined Train": "./data/kaleel_final_dataset_train_Combined_Grayscale.pth",
}

DATASETS = {**VAL_DATASETS, **TRAIN_DATASETS}

# ------------------------ Scanned Bubble Datasets (Post Print-Scan) ------------------------------
SCANNED_DATASETS_DIR = "./data"

SCANNED_DATASETS = {
    "resnet20_combined": f"{SCANNED_DATASETS_DIR}/ResNet-Combined_correct_samples_1000_scanned_bubbles.pt",
    "cait_combined": f"{SCANNED_DATASETS_DIR}/CaiT-Combined_correct_samples_1000_scanned_bubbles.pt",
    "vgg16_combined": f"{SCANNED_DATASETS_DIR}/VGG-Combined_correct_samples_1000_scanned_bubbles.pt",
    "svm_combined": f"{SCANNED_DATASETS_DIR}/SVM-Combined_correct_samples_1000_scanned_bubbles.pt",
    "snn_vgg16_combined": f"{SCANNED_DATASETS_DIR}/SNN-VGG-Combined_correct_samples_1000_scanned_bubbles.pt",
    "snn_resnet20_combined": f"{SCANNED_DATASETS_DIR}/SNN-ResNet-Combined_correct_samples_1000_scanned_bubbles.pt",
    "expv2_vgg16": f"{SCANNED_DATASETS_DIR}/xAI_VGG16-Combined_correct_samples_1000_scanned_bubbles.pt",
    "expv2_resnet20": f"{SCANNED_DATASETS_DIR}/xAI-ResNet20-Combined_correct_samples_1000_scanned_bubbles.pt",
    "mambavision_combined": f"{SCANNED_DATASETS_DIR}/MambaVision-L2-Combined_correct_samples_1000_scanned_bubbles.pt",
}

# ========================================================================================
# Individual Experiments (Without UNet) - Uses Original Validation Data
# ========================================================================================

EXPERIMENTS_RESNET20 = {
    "resnet20_combined": {
        "ckpt_path": CHECKPOINTS["resnet20_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_VGG16 = {
    "vgg16_combined": {
        "ckpt_path": CHECKPOINTS["vgg16_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_SVM = {
    "svm_combined": {
        "ckpt_path": CHECKPOINTS["svm_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_CAIT = {
    "cait_combined": {
        "ckpt_path": CHECKPOINTS["cait_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_SNN_VGG16 = {
    "snn_vgg16_combined": {
        "ckpt_path": CHECKPOINTS["snn_vgg16_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_SNN_RESNET20 = {
    "snn_resnet20_combined": {
        "ckpt_path": CHECKPOINTS["snn_resnet20_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_XAI_RESNET20 = {
    "expv2_resnet20": {
        "ckpt_path": CHECKPOINTS["expv2_resnet20"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_XAI_VGG16 = {
    "expv2_vgg16": {
        "ckpt_path": CHECKPOINTS["expv2_vgg16"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

EXPERIMENTS_MAMBAVISION = {
    "mambavision_combined": {
        "ckpt_path": CHECKPOINTS["mambavision_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

# ========================================================================================
# Individual Experiments (With UNet) - Uses Scanned Bubble Data
# ========================================================================================

EXPERIMENTS_UNET_RESNET20 = {
    "resnet20_combined": {
        "ckpt_path": CHECKPOINTS["resnet20_combined"],
        "dataset_path": SCANNED_DATASETS["resnet20_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "ResNet20-C",
    },
}

EXPERIMENTS_UNET_VGG16 = {
    "vgg16_combined": {
        "ckpt_path": CHECKPOINTS["vgg16_combined"],
        "dataset_path": SCANNED_DATASETS["vgg16_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "VGG16-C",
    },
}

EXPERIMENTS_UNET_SVM = {
    "svm_combined": {
        "ckpt_path": CHECKPOINTS["svm_combined"],
        "dataset_path": SCANNED_DATASETS["svm_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "SVM-C",
    },
}

EXPERIMENTS_UNET_CAIT = {
    "cait_combined": {
        "ckpt_path": CHECKPOINTS["cait_combined"],
        "dataset_path": SCANNED_DATASETS["cait_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "CaiT-C",
    },
}

EXPERIMENTS_UNET_SNN_VGG16 = {
    "snn_vgg16_combined": {
        "ckpt_path": CHECKPOINTS["snn_vgg16_combined"],
        "dataset_path": SCANNED_DATASETS["snn_vgg16_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "SNN_VGG16-C",
    },
}

EXPERIMENTS_UNET_SNN_RESNET20 = {
    "snn_resnet20_combined": {
        "ckpt_path": CHECKPOINTS["snn_resnet20_combined"],
        "dataset_path": SCANNED_DATASETS["snn_resnet20_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "SNN_ResNet20-C",
    },
}

EXPERIMENTS_UNET_XAI_RESNET20 = {
    "expv2_resnet20": {
        "ckpt_path": CHECKPOINTS["expv2_resnet20"],
        "dataset_path": SCANNED_DATASETS["expv2_resnet20"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "Explainable_AI_ResNet20-C",
    },
}

EXPERIMENTS_UNET_XAI_VGG16 = {
    "expv2_vgg16": {
        "ckpt_path": CHECKPOINTS["expv2_vgg16"],
        "dataset_path": SCANNED_DATASETS["expv2_vgg16"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "Explainable_AI_VGG16-C",
    },
}

EXPERIMENTS_UNET_MAMBAVISION = {
    "mambavision_combined": {
        "ckpt_path": CHECKPOINTS["mambavision_combined"],
        "dataset_path": SCANNED_DATASETS["mambavision_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "MambaVision-L2-C",
    },
}

# ========================================================================================
# All Models Combined (Without UNet) - Uses Original Validation Data
# ========================================================================================

EXPERIMENTS_ALL = {
    "resnet20_combined": {
        "ckpt_path": CHECKPOINTS["resnet20_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "cait_combined": {
        "ckpt_path": CHECKPOINTS["cait_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "vgg16_combined": {
        "ckpt_path": CHECKPOINTS["vgg16_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "svm_combined": {
        "ckpt_path": CHECKPOINTS["svm_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "snn_vgg16_combined": {
        "ckpt_path": CHECKPOINTS["snn_vgg16_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "snn_resnet20_combined": {
        "ckpt_path": CHECKPOINTS["snn_resnet20_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "expv2_vgg16": {
        "ckpt_path": CHECKPOINTS["expv2_vgg16"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "expv2_resnet20": {
        "ckpt_path": CHECKPOINTS["expv2_resnet20"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
    "mambavision_combined": {
        "ckpt_path": CHECKPOINTS["mambavision_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
        "use_unet": False,
    },
}

# ========================================================================================
# All Models Combined (With UNet) - Uses Scanned Bubble Data
# ========================================================================================

EXPERIMENTS_UNET_ALL = {
    "resnet20_combined": {
        "ckpt_path": CHECKPOINTS["resnet20_combined"],
        "dataset_path": SCANNED_DATASETS["resnet20_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "ResNet20-C",
    },
    "cait_combined": {
        "ckpt_path": CHECKPOINTS["cait_combined"],
        "dataset_path": SCANNED_DATASETS["cait_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "CaiT-C",
    },
    "vgg16_combined": {
        "ckpt_path": CHECKPOINTS["vgg16_combined"],
        "dataset_path": SCANNED_DATASETS["vgg16_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "VGG16-C",
    },
    "svm_combined": {
        "ckpt_path": CHECKPOINTS["svm_combined"],
        "dataset_path": SCANNED_DATASETS["svm_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "SVM-C",
    },
    "snn_vgg16_combined": {
        "ckpt_path": CHECKPOINTS["snn_vgg16_combined"],
        "dataset_path": SCANNED_DATASETS["snn_vgg16_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "SNN_VGG16-C",
    },
    "snn_resnet20_combined": {
        "ckpt_path": CHECKPOINTS["snn_resnet20_combined"],
        "dataset_path": SCANNED_DATASETS["snn_resnet20_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "SNN_ResNet20-C",
    },
    "expv2_vgg16": {
        "ckpt_path": CHECKPOINTS["expv2_vgg16"],
        "dataset_path": SCANNED_DATASETS["expv2_vgg16"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "Explainable_AI_VGG16-C",
    },
    "expv2_resnet20": {
        "ckpt_path": CHECKPOINTS["expv2_resnet20"],
        "dataset_path": SCANNED_DATASETS["expv2_resnet20"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "Explainable_AI_ResNet20-C",
    },
    "mambavision_combined": {
        "ckpt_path": CHECKPOINTS["mambavision_combined"],
        "dataset_path": SCANNED_DATASETS["mambavision_combined"],
        "use_unet": True,
        "unet_ckpt": UNET_CHECKPOINT,
        "display_name": "MambaVision-L2-C",
    },
}