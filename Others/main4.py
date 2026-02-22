'''
Standard main function to load the model, do adversarial attack and save adversarial sample and adversarial images
'''
import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from model_architecture import ResNet, cait, VGG, MultiOutputSVM
import rays_attack
import surfree_attack
import square_attack
import adba_attack
import utils

import matplotlib.pyplot as plt
import os
from PIL import Image

def main():

    # --------- ResNet models ---------
    modelDir="./checkpoint/ModelResNet20-VotingOnlyBubbles-v2-Grayscale-Run1.th"
    # modelDir = "./checkpoint/ModelResNet20-VotingCombined-v2-Grayscale-Run1.th"
    
    # --------- CaiT models ----------
    # modelDir = "./checkpoint/ModelCaiT-VotingOnlyBubbles-v2-Grayscale-Run1.th"
    # modelDir = "./checkpoint/ModelCaiT-trCombined-v2-valCombined-v2-Grayscale-Run1.th"
    
    # --------- VGG models -----------
    # modelDir = "./checkpoint/ModelVgg16-B.th"
    # modelDir = "./checkpoint/ModelVgg16-C2.th"
    
    # --------- SVM models ------------
    # # ---- OnlyBubbles ----
    # modelDir_base  = "./checkpoint/sklearn_SVM_OnlyBubbles_v2_Grayscale_Run1/base_pytorch_svm_OnlyBubbles_v2.pth"
    # modelDir_multi = "./checkpoint/sklearn_SVM_OnlyBubbles_v2_Grayscale_Run1/multi_output_svm_OnlyBubbles_v2.pth"
    # ---- Combined ----
    # modelDir_base  = "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/base_pytorch_svm_combined_v2.pth"
    # modelDir_multi = "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/multi_output_svm_combined_v2.pth"

    # # Parameters for the dataset
    batchSize = 64
    numClasses = 2
    inputImageSize = [1, 1, 40, 50]
    dropOutRate = 0.0

    # Define GPU device
    device = torch.device("cuda")

    # -------------- Loading ResNet model ------------------
    # Create the ResNet model
    model = ResNet.resnet20(inputImageSize, dropOutRate, numClasses).to(device)
    # Load in the trained weights of the model
    checkpoint = torch.load(modelDir, weights_only=False)
    # Load the state dictionary into the model
    model.load_state_dict(checkpoint["state_dict"])

    # -------------- Loading CaiT model ---------------------
    # model = cait.CaiT(
    #     image_size   =(40, 50),
    #     patch_size   =5,
    #     num_classes  =2,
    #     num_channels =1,
    #     dim          =512,
    #     depth        =16,
    #     cls_depth    =2,
    #     heads        =8,
    #     mlp_dim      =2048,
    #     dropout      =0.1,
    #     emb_dropout  =0.1,
    #     layer_dropout=0.05
    # ).to(device)

    # ckpt = torch.load(modelDir, map_location=device, weights_only=False)
    # model.load_state_dict(ckpt["state_dict"])
    
    # # IMPORTANT: CaiT drops layers even in eval(); disable for deterministic eval/attack
    # model.patch_transformer.layer_dropout = 0.0
    # model.cls_transformer.layer_dropout   = 0.0

    # ----------------- Loading VGG model ----------------------
    # imgH, imgW   = 40, 50
    # # Create the VGG16 model
    # model = VGG.VGG("VGG16", imgH, imgW, numClasses).to(device)
    # # Load checkpoint (handles raw state_dict or dict with 'state_dict'; strips 'module.' if present)
    # raw = torch.load(modelDir, map_location="cpu")
    # state = raw.get("state_dict", raw)
    # state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    
    # missing, unexpected = model.load_state_dict(state, strict=False)
    # print(f"Loaded checkpoint (strict=False) | missing={len(missing)} unexpected={len(unexpected)}")

    # ----------------- Loading SVM model ------------------------
    # INPUT_DIM = 1 * 40 * 50  # 2000

    # base_state = torch.load(modelDir_base, map_location="cpu")
    # model = MultiOutputSVM.MultiOutputSVM(INPUT_DIM, base_state).to(device)
    
    # multi_state = torch.load(modelDir_multi, map_location="cpu")
    # model.load_state_dict(multi_state)

    # ------------------------------------------------------------

    # Switch the model into eval model for testing
    model= model.eval()
    
    # ---------------- dataset --------------
    # data = torch.load("./data/kaleel_final_dataset_train_OnlyBubbles_Grayscale.pth", weights_only=False)
    data = torch.load("./data/kaleel_final_dataset_val_OnlyBubbles_Grayscale.pth", weights_only=False)
    # data = torch.load("./data/kaleel_final_dataset_train_Combined_Grayscale.pth", weights_only=False)
    # data = torch.load("./data/kaleel_final_dataset_val_Combined_Grayscale.pth", weights_only=False)
    images = data["data"].float()
    labels_binary = data["binary_labels"].long()
    labels_original = data["original_labels"].long()

    print(f"Image shape: {images.shape}")

    # Create a dataloader with only images and binary labels
    dataset = TensorDataset(images, labels_binary)
    valLoader = DataLoader(dataset, batch_size = batchSize, shuffle = False)

    # Check the clean accuracy of the model
    cleanAcc = utils.validateD(valLoader, model, device)
    print("Voter Dataset Val Loader Clean Acc:", cleanAcc)

    # Get correctly classified, classwise balanced samples to do the attack
    totalSamplesRequired = 1000
    correctLoader = utils.GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, valLoader, numClasses)

    # Check the number of samples in the correctLoader
    print("Number of samples in correctLoader:", len(correctLoader.dataset))

    # Correct Classifier Accuracy
    corrAcc = utils.validateD(correctLoader, model, device)
    print("Voter Dataset correctLoader Clean Acc:", corrAcc)

    # # Get pixel bounds for a correctLoader
    minVal, maxVal = utils.GetDataBounds(correctLoader, device)
    print("Data Range for Correct Loader:", [round(minVal, 4), round(maxVal, 4)])

    # -------------- Linf Attack - RaySAttack ------------
    # epsilonMax = 255/255
    # queryLimit = 10000
    
    # advLoaderRayS = rays_attack.RaySAttack(device, model, epsilonMax, queryLimit, correctLoader)
    # advAcc = utils.validateD(advLoaderRayS, model, device)
    # print("RayS black-box robustness:", advAcc)

    # ----------------------- L2 Attack - Surfree Attack ---------------
    
    # # Configuration for SurFree attack
    # surfree_config = {
    #     "init": {
    #         "steps": 100,                       # Number of optimization steps
    #         "max_queries": 500,                # Maximum queries per image
    #         "BS_gamma": 0.01,                   # Binary search precision threshold
    #         "BS_max_iteration": 7,              # Max binary search iterations
    #         "theta_max": 30,                    # Max angle for direction search (degrees)
    #         "n_ortho": 100,                     # Number of orthogonal directions to maintain
    #         "rho": 0.95,                        # Angle adjustment factor
    #         "T": 1,                             # Number of evaluations per direction
    #         "with_alpha_line_search": True,     # Enable binary search on theta
    #         "with_distance_line_search": False, # Enable binary search on distance
    #         "with_interpolation": False,        # Enable interpolation
    #         "final_line_search": True,          # Perform final line search
    #         "quantification": True,             # Quantify to valid pixel values
    #         "clip": True                        # Clip values to [0, 1]
    #     },
    #     "run": {
    #         "basis_params": {
    #             "basis_type": "random",            # Type of basis ("dct" or "random")
    #             "dct_type": "full",             # DCT type ("full" or "8x8")
    #             "frequence_range": (0, 0.5),    # Frequency range for DCT
    #             "beta": 0.001,                  # Noise factor for DCT basis
    #             "tanh_gamma": 1,                # Gamma for tanh function in DCT
    #             "random_noise": "normal"        # Noise type ("normal" or "uniform")
    #         }
    #     }
    # }

    # Configuration for SurFree attack
    surfree_config = {
        "init": {
            "steps": 100,                       # Number of optimization steps
            "max_queries": 10000,                # Maximum queries per image
            "BS_gamma": 0.001,                   # Binary search precision threshold
            "BS_max_iteration": 10,              # Max binary search iterations
            "theta_max": 30,                    # Max angle for direction search (degrees)
            "n_ortho": 100,                     # Number of orthogonal directions to maintain
            "rho": 0.95,                        # Angle adjustment factor
            "T": 1,                             # Number of evaluations per direction
            "with_alpha_line_search": True,     # Enable binary search on theta
            "with_distance_line_search": False, # Enable binary search on distance
            "with_interpolation": False,        # Enable interpolation
            "final_line_search": True,          # Perform final line search
            "quantification": True,             # Quantify to valid pixel values
            "clip": True                        # Clip values to [0, 1]
        },
        "run": {
            "basis_params": {
                "basis_type": "random",            # Type of basis ("dct" or "random")
                "dct_type": "full",             # DCT type ("full" or "8x8")
                "frequence_range": (0, 0.5),    # Frequency range for DCT
                "beta": 0.001,                  # Noise factor for DCT basis
                "tanh_gamma": 1,                # Gamma for tanh function in DCT
                "random_noise": "normal"        # Noise type ("normal" or "uniform")
            }
        }
    }
    
    # Run SurFree attack
    advLoaderSurFree, blurb = surfree_attack.SurFree_AttackWrapper(
        model=model,
        device=device,
        dataLoader=correctLoader,
        config=surfree_config,
    )
    
    # Calculate adversarial accuracy
    advAcc = utils.validateD(advLoaderSurFree, model, device)
    print("Surfree L2 black-box robustness:", advAcc)


    # ---------------- Square Attack L-inf ------------------
    # # Untargeted Square Attack L-inf
    # advLoaderSquareLinf = square_attack.SquareAttackLinf_Wrapper(
    #     model=model,
    #     device=device,
    #     dataLoader=correctLoader,
    #     eps=4/255,
    #     n_iters=40000,
    #     p_init=0.8,
    #     n_classes=numClasses,
    #     targeted=False,
    #     loss_type = "cross_entropy"
    # )
    
    # advAccSquareLinf = utils.validateD(advLoaderSquareLinf, model, device)
    # print("Untargeted Square Attack Linf black-box robustness:", advAccSquareLinf)

    # --------
    # Targeted Square Attack L-inf
    # advLoaderSquareLinf = square_attack.SquareAttackLinf_Wrapper(
    #     model=model,
    #     device=device,
    #     dataLoader=correctLoader,
    #     eps=1,
    #     n_iters=25000,
    #     p_init=0.8,
    #     n_classes=numClasses,
    #     targeted=True,
    #     loss_type = "cross_entropy",
    # )
    
    # advAccSquareLinf = utils.validateD(advLoaderSquareLinf, model, device)
    # print("Targeted Square Attack Linf black-box robustness:", advAccSquareLinf)

    # ----------------- Square Attack l2 -----------------------
    # advLoaderSquareL2 = square_attack.SquareAttackL2_Wrapper(
    #     model=model,
    #     device=device,
    #     dataLoader=correctLoader,
    #     eps=45,
    #     n_iters=40000,
    #     p_init=0.02,
    #     n_classes=numClasses,
    #     targeted=False,
    # )
    
    # advAccSquareL2 = utils.validateD(advLoaderSquareL2, model, device)
    # print("Square Attack L2 black-box robustness:", advAccSquareL2)

    # ----------------------- L-inf Attack - ADBA Attack ---------------
    
    # Configuration for ADBA attack
    # adba_config = {
    #     "epsilon": 255/255,          # L-inf perturbation bound (adjust as needed)
    #     "budget": 10000,           # Maximum queries per image
    #     "init_dir": 1,            # Initial direction: 0=random, 1=all +1, -1=all -1
    #     "offspring_n": 2,         # Number of offspring directions
    #     "binary_mode": 0,          # Binary search mode: 0=midpoint, 1=median-based
    #     "channels": 1             # Grayscale images
    # }

    # print(f"\n{'=' * 60}\nStarting ADBA Attack...\nConfig: {adba_config}\n{'=' * 60}")
    
    # # Run ADBA attack
    # adba_advLoader = adba_attack.ADBA_AttackWrapper(
    #     model=model,
    #     device=device,
    #     dataLoader=correctLoader,
    #     config=adba_config
    # )
    
    # # Calculate adversarial accuracy
    # adba_advAcc = utils.validateD(adba_advLoader, model, device)
    # print("ADBA Attack Linf black-box robustness:", adba_advAcc)

    # -------------- Save Adversarial Loader as Dictionary .pth ----------------

    # # Create output directory
    # output_dir = "./adversarial_samples"
    # os.makedirs(output_dir, exist_ok=True)
    
    # # Extract all adversarial images and labels from the loader
    # adv_images = []
    # adv_labels = []
    
    # for batch_images, batch_labels in advLoaderSquareLinf:
    #     adv_images.append(batch_images.cpu())
    #     adv_labels.append(batch_labels.cpu())
    
    # # Concatenate all batches into single tensors
    # x_adv = torch.cat(adv_images, dim=0).float()  # [1000, 1, 40, 50], float32 in [0,1]
    # y_clean = torch.cat(adv_labels, dim=0).long()  # 0/1 labels
    
    # print(f"\nAdversarial data shapes:")
    # print(f"  - x_adv: {x_adv.shape}")
    # print(f"  - y_clean: {y_clean.shape}")
    # print(f"  - x_adv range: [{x_adv.min():.4f}, {x_adv.max():.4f}]")
    
    # # Create the adversarial blob dictionary
    # adv_blob = {
    #     "data": x_adv,  # [1000, 1, 40, 50], float32 in [0,1]
    #     "binary_labels": y_clean,  # 0/1
    #     "original_labels": y_clean,  # same as binary_labels for OnlyBubbles
    # }
    
    # # Save to a .pth file
    # save_path = "./adversarial_samples/Square-Linf_adv_SVM-B_eps4by255.pth"
    # torch.save(adv_blob, save_path)
    # print(f"\n Adversarial blob saved to: {save_path}")
    # print(f"  Dictionary keys: {list(adv_blob.keys())}")

    # # ------------- Save 5 adversarial samples ---------------

    # # Create output directory
    # output_imgdir = "./adversarial_images"
    # os.makedirs(output_imgdir, exist_ok=True)
    
    # # Track how many images saved per label
    # saved_count = {0: 0, 1: 0}
    # target_count = 10
    
    # # Iterate through adversarial loader
    # for adv_images, labels in advLoaderSurFree:
    #     for i in range(len(labels)):
    #         label = labels[i].item()
            
    #         # Check if we still need images for this label
    #         if saved_count[label] < target_count:
    #             # Convert tensor to numpy array
    #             img = adv_images[i].squeeze().cpu().numpy()
                
    #             # Scale to 0-255 and convert to uint8
    #             img = (img * 255).astype(np.uint8)
                
    #             # Save as image
    #             pil_img = Image.fromarray(img, mode='L')  # 'L' for grayscale
    #             pil_img.save(f"{output_imgdir}/adv_label_{label}_sample_{saved_count[label]}.png")
                
    #             saved_count[label] += 1
            
    #         # Stop if we have enough for both labels
    #         if saved_count[0] >= target_count and saved_count[1] >= target_count:
    #             break
        
    #     if saved_count[0] >= target_count and saved_count[1] >= target_count:
    #         break
    
    # print(f"Saved {saved_count[0]} images for label 0")
    # print(f"Saved {saved_count[1]} images for label 1")

    

if __name__ == '__main__':
    main()