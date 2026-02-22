'''
Standard main function to load the model and do adversarial attack and print robust accuracy
'''

import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader

from model_architecture import ResNet, cait, VGG, MultiOutputSVM
import utils

import APGDOriginal  

def main():

    # --------- ResNet models ---------
    # modelDir="./checkpoint/ModelResNet20-VotingOnlyBubbles-v2-Grayscale-Run1.th"
    # modelDir = "./checkpoint/ModelResNet20-VotingCombined-v2-Grayscale-Run1.th"
    
    # --------- CaiT models ----------
    # modelDir = "./checkpoint/ModelCaiT-VotingOnlyBubbles-v2-Grayscale-Run1.th"
    # modelDir = "./checkpoint/ModelCaiT-trCombined-v2-valCombined-v2-Grayscale-Run1.th"
    
    # --------- VGG models -----------
    modelDir = "./checkpoint/ModelVgg16-B.th"
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
    # # Create the ResNet model
    # model = ResNet.resnet20(inputImageSize, dropOutRate, numClasses).to(device)
    # # Load in the trained weights of the model
    # checkpoint = torch.load(modelDir, weights_only=False)
    # # Load the state dictionary into the model
    # model.load_state_dict(checkpoint["state_dict"])

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
    imgH, imgW   = 40, 50
    # Create the VGG16 model
    model = VGG.VGG("VGG16", imgH, imgW, numClasses).to(device)
    # Load checkpoint (handles raw state_dict or dict with 'state_dict'; strips 'module.' if present)
    raw = torch.load(modelDir, map_location="cpu")
    state = raw.get("state_dict", raw)
    state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint (strict=False) | missing={len(missing)} unexpected={len(unexpected)}")

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

    # -------------- Linf Attack - APGD (White-Box) ------------
    epsilonMax = 8/255          # L-inf perturbation bound
    etaStart = 3 * epsilonMax   # Initial step size (typically 2x epsilon)
    numSteps = 100              # Number of attack iterations
    clipMin = 0.0               # Minimum pixel value
    clipMax = 1.0               # Maximum pixel value
    
    print(f"\n{'=' * 60}")
    print(f"Running APGD Attack...")
    print(f"  Epsilon: {epsilonMax:.4f} ({int(epsilonMax * 255)}/255)")
    print(f"  Eta (step size): {etaStart:.4f}")
    print(f"  Num Steps: {numSteps}")
    print(f"  Clip Range: [{clipMin}, {clipMax}]")
    print(f"{'=' * 60}")
    
    advLoaderAPGD = APGDOriginal.AutoAttackPytorchMatGPUWrapper(
        device, 
        correctLoader, 
        model, 
        epsilonMax, 
        etaStart, 
        numSteps, 
        clipMin, 
        clipMax
    )
    
    advAccAPGD = utils.validateD(advLoaderAPGD, model, device)
    print(f"\nAPGD Linf white-box robustness (eps={int(epsilonMax * 255)}/255): {advAccAPGD:.4f}")

    # ------------- Check accuracy of advLoaderAPGD with Resnet-B, Resnet-C, VGG-B, VGG-C, SVM-B, SVM-C, CAIT-B, CAIT-C

if __name__ == '__main__':
    main()