# AdaptiveAttack.py

"""
Adaptive Black Box Attack - Compatible with Model Only AND UNet+Model Mode

This script attacks oracle models using a synthetic VGG11 model for transfer.
Supports both:
  - Model Only: Uses original validation data for attack
  - UNet+Model: Uses scanned bubble data for attack (with UNet denoiser)
  
Training always uses voter training data.
"""

import torch
import random
import os
from torch.utils.data import DataLoader, TensorDataset

from ModelFactory import ModelFactory
import AttackWrappersAdaptiveBlackBox
import utils
from constants import (
    CHECKPOINTS,
    UNET_CHECKPOINT,
    EXPERIMENTS_ALL,
    EXPERIMENTS_UNET_ALL,
    EXPERIMENTS_SNN_RESNET20
)

# ------ Setting seed for reproducibility -------
SEED = 20

def reset_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed at module load
reset_seed()

# ------- SYTHETIC MODEL AND ORACLE CONFIGURATION ------------

# Synthetic model for transfer attack (VGG11 - will be trained, no checkpoint needed)
SYNTHETIC_MODEL = "vgg11"

# Choose experiment mode: "model_only" or "unet"
ATTACK_MODE = "model_only"  # for Model only attacks
# ATTACK_MODE = "unet"        # for UNet+Model attacks

# For Attack mode = model_only
EXPERIMENTS_MODEL_ONLY = EXPERIMENTS_ALL

# For Attack mode = unet (UNet+Model attacks)
# EXPERIMENTS_UNET_MODE = EXPERIMENTS_UNET_ALL

# -------- ATTACK FUNCTION -----------

def run_attack_on_oracle(model_name: str, config: dict, synthetic_model_name: str):
    """
    Run adaptive black box attack on a single oracle model.
    
    Args:
        model_name: Name of the oracle model
        config: Configuration dict with ckpt_path, dataset_path, use_unet, etc.
        synthetic_model_name: Name of synthetic model for transfer (e.g., "vgg11")
    """

    # ------ Seed is reset in each indvidual oracle model initiation for reproducibility -----
    reset_seed(SEED)
    
    # Parameters
    batchSize = 64
    numClasses = 2
    numTrainingSamples = 2000
    numAttackSamples = 500
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract config
    use_unet = config.get("use_unet", False)
    model_ckpt = config["ckpt_path"]
    dataset_path = config["dataset_path"]  # Scanned data path for UNet mode
    unet_ckpt = config.get("unet_ckpt", UNET_CHECKPOINT) if use_unet else None
    
    mode_str = "UNet+Model" if use_unet else "Model Only"
    
    print(f"Oracle model: {model_name} ({mode_str})", flush=True)
    print(f"Synthetic model: {synthetic_model_name} (will be trained)", flush=True)
    
    # Load models
    factory = ModelFactory(device=device)
    
    # Load synthetic model (VGG11 - untrained, no checkpoint)
    syntheticModel = factory.get_model(synthetic_model_name)
    
    # Load oracle model (with or without UNet wrapper)
    if use_unet:
        oracle = factory.get_unet_model_wrapper(
            model_name=model_name,
            model_checkpoint=model_ckpt,
            unet_checkpoint=unet_ckpt,
        )
    else:
        oracle = factory.get_model(model_name, model_ckpt)
    
    # Create save tag
    unet_suffix = "_unet" if use_unet else ""
    saveTag = f"Adaptive_Attack_Oracle={model_name}{unet_suffix}_Synthetic={synthetic_model_name}"
    
    #-------------- DATA LOADERS ---------------
    
    # Training & Validation Dataloaders
    trainLoader = utils.GetVoterTrainingBalanced(batchSize, numTrainingSamples, numClasses)
    valLoader = utils.GetVoterValidation(batchSize)
    
    # Attack dataloder
    if use_unet:
        # UNet+Model: Use scanned bubble data for attack
        attackLoader = utils.get_scanned_attack_loader(dataset_path, batchSize)
    else:
        # Model Only: Use valLoader data for attack
        attackLoader = utils.GetVoterValidation(batchSize) 
    
    # Training config
    training_config = {
        "batchSize": batchSize,
        "numIterations": 4,    
        "epochsPerIteration": 10,
        "epsForAug": 0.01,
        "learningRate": 0.0001,
        "numTrainingSamples": numTrainingSamples,
        "dataLoaderForTraining": trainLoader,
        "valLoader": valLoader,
        "optimizerName": "adam",
        "clipMin": 0.0,
        "clipMax": 1.0,
    }
    
    # Run attack
    try:
        AttackWrappersAdaptiveBlackBox.AdaptiveAttack(
            saveTag=saveTag,
            device=device,
            oracle=oracle,
            syntheticModel=syntheticModel,
            numClasses=numClasses,
            training_config=training_config,
            numAttackSamples=numAttackSamples,
            attackLoader=attackLoader,
        )
        print(f"Attack completed for {model_name}", flush=True)
        return True
    except Exception as e:
        print(f"Attack failed for {model_name}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def main():
    # Select experiments based on mode
    if ATTACK_MODE == "unet":
        experiments = EXPERIMENTS_UNET_MODE
        mode_name = "UNet+Model"
        results_file = "adaptive_unet_results.txt"
    else:
        experiments = EXPERIMENTS_MODEL_ONLY
        mode_name = "Model Only"
        results_file = "adaptive_results.txt"
    
    results = {}
    
    for model_name, config in experiments.items():
        success = run_attack_on_oracle(
            model_name=model_name,
            config=config,
            synthetic_model_name=SYNTHETIC_MODEL,
        )
        results[model_name] = "Success" if success else "Failed"
    
    # Save results
    with open(results_file, "a", encoding="utf-8") as f:
        header = f" ADAPTIVE BLACK BOX ATTACK ({mode_name}) RESULTS "
        f.write("\n" + "=" * 10 + header + "=" * 10 + "\n")
        for model, status in results.items():
            use_unet = experiments[model].get("use_unet", False)
            unet_suffix = "_unet" if use_unet else ""
            f.write(f"{model}{unet_suffix:<40}: {status}\n")


if __name__ == '__main__':
    main()