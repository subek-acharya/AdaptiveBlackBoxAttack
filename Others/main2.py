'''
This code check attack transferability from source model to other models
'''

import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader

from model_architecture import ResNet, cait, VGG, MultiOutputSVM
import utils

import APGDOriginal  

def load_resnet(modelDir, device, inputImageSize, dropOutRate, numClasses):
    """Load ResNet model"""
    model = ResNet.resnet20(inputImageSize, dropOutRate, numClasses).to(device)
    checkpoint = torch.load(modelDir, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def load_vgg(modelDir, device, imgH, imgW, numClasses):
    """Load VGG model"""
    model = VGG.VGG("VGG16", imgH, imgW, numClasses).to(device)
    raw = torch.load(modelDir, map_location="cpu")
    state = raw.get("state_dict", raw)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def load_cait(modelDir, device):
    """Load CaiT model"""
    model = cait.CaiT(
        image_size=(40, 50),
        patch_size=5,
        num_classes=2,
        num_channels=1,
        dim=512,
        depth=16,
        cls_depth=2,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        layer_dropout=0.05
    ).to(device)
    
    ckpt = torch.load(modelDir, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    
    # Disable layer dropout for deterministic evaluation
    model.patch_transformer.layer_dropout = 0.0
    model.cls_transformer.layer_dropout = 0.0
    model.eval()
    return model

def load_svm(modelDir_base, modelDir_multi, device):
    """Load SVM model"""
    INPUT_DIM = 1 * 40 * 50  # 2000
    
    base_state = torch.load(modelDir_base, map_location="cpu")
    model = MultiOutputSVM.MultiOutputSVM(INPUT_DIM, base_state).to(device)
    
    multi_state = torch.load(modelDir_multi, map_location="cpu")
    model.load_state_dict(multi_state)
    model.eval()
    return model


def main():

    # ===================== MODEL PATHS =====================
    # ResNet models
    resnet_B_path = "./checkpoint/ModelResNet20-VotingOnlyBubbles-v2-Grayscale-Run1.th"
    resnet_C_path = "./checkpoint/ModelResNet20-VotingCombined-v2-Grayscale-Run1.th"
    
    # CaiT models
    cait_B_path = "./checkpoint/ModelCaiT-VotingOnlyBubbles-v2-Grayscale-Run1.th"
    cait_C_path = "./checkpoint/ModelCaiT-trCombined-v2-valCombined-v2-Grayscale-Run1.th"
    
    # VGG models
    vgg_B_path = "./checkpoint/ModelVgg16-B.th"
    vgg_C_path = "./checkpoint/ModelVgg16-C2.th"
    
    # SVM models
    svm_B_base = "./checkpoint/sklearn_SVM_OnlyBubbles_v2_Grayscale_Run1/base_pytorch_svm_OnlyBubbles_v2.pth"
    svm_B_multi = "./checkpoint/sklearn_SVM_OnlyBubbles_v2_Grayscale_Run1/multi_output_svm_OnlyBubbles_v2.pth"
    svm_C_base = "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/base_pytorch_svm_combined_v2.pth"
    svm_C_multi = "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/multi_output_svm_combined_v2.pth"

    # ===================== PARAMETERS =====================
    batchSize = 64
    numClasses = 2
    inputImageSize = [1, 1, 40, 50]
    dropOutRate = 0.0
    imgH, imgW = 40, 50

    # Define GPU device
    device = torch.device("cuda")

    # ===================== LOAD SOURCE MODEL (VGG-B) =====================
    print("=" * 70)
    print("Loading Source Model: VGG-B")
    print("=" * 70)
    
    source_model = load_vgg(vgg_B_path, device, imgH, imgW, numClasses)
    print("VGG-B loaded successfully")

    # ===================== LOAD DATASET =====================
    data = torch.load("./data/kaleel_final_dataset_train_OnlyBubbles_Grayscale.pth", weights_only=False)
    images = data["data"].float()
    labels_binary = data["binary_labels"].long()

    print(f"Image shape: {images.shape}")

    dataset = TensorDataset(images, labels_binary)
    valLoader = DataLoader(dataset, batch_size=batchSize, shuffle=False)

    # Check clean accuracy of source model
    cleanAcc = utils.validateD(valLoader, source_model, device)
    print(f"Source Model (VGG-B) Clean Acc: {cleanAcc:.4f}")

    # Get correctly classified samples
    totalSamplesRequired = 1000
    correctLoader = utils.GetCorrectlyIdentifiedSamplesBalanced(
        source_model, totalSamplesRequired, valLoader, numClasses
    )
    print(f"Number of samples in correctLoader: {len(correctLoader.dataset)}")

    # # ===================== GENERATE ADVERSARIAL EXAMPLES =====================
    # epsilonMax = 8/255
    # etaStart = 2 * epsilonMax
    # numSteps = 500
    # clipMin = 0.0
    # clipMax = 1.0
    
    # print(f"\n{'=' * 70}")
    # print(f"Generating APGD Adversarial Examples (Source: VGG-B)")
    # print(f"{'=' * 70}")
    # print(f"  Epsilon: {epsilonMax:.4f} ({int(epsilonMax * 255)}/255)")
    # print(f"  Eta (step size): {etaStart:.4f}")
    # print(f"  Num Steps: {numSteps}")
    
    # advLoaderAPGD = APGDOriginal.DLR_AutoAttackPytorchMatGPUWrapper(
    #     device, 
    #     correctLoader, 
    #     source_model, 
    #     epsilonMax, 
    #     etaStart, 
    #     numSteps, 
    #     clipMin, 
    #     clipMax
    # )
    
    # # ===================== EVALUATE TRANSFERABILITY =====================
    # print(f"\n{'=' * 70}")
    # print(f"TRANSFERABILITY RESULTS")
    # print(f"Adversarial examples generated using: VGG-B")
    # print(f"Epsilon: {int(epsilonMax * 255)}/255")
    # print(f"{'=' * 70}")
    
    # results = {}
    
    # # ------------- VGG-B (Source Model) -------------
    # print("\n--- VGG-B (Source Model) ---")
    # acc = utils.validateD(advLoaderAPGD, source_model, device)
    # results["VGG-B"] = acc
    # print(f"  Robust Accuracy: {acc:.4f}")
    
    # # Clean up source model to free GPU memory
    # del source_model
    # torch.cuda.empty_cache()
    
    # # ------------- VGG-C -------------
    # print("\n--- VGG-C ---")
    # model_vgg_c = load_vgg(vgg_C_path, device, imgH, imgW, numClasses)
    # acc = utils.validateD(advLoaderAPGD, model_vgg_c, device)
    # results["VGG-C"] = acc
    # print(f"  Robust Accuracy: {acc:.4f}")
    # del model_vgg_c
    # torch.cuda.empty_cache()
    
    # # ------------- ResNet-B -------------
    # print("\n--- ResNet-B ---")
    # model_resnet_b = load_resnet(resnet_B_path, device, inputImageSize, dropOutRate, numClasses)
    # acc = utils.validateD(advLoaderAPGD, model_resnet_b, device)
    # results["ResNet-B"] = acc
    # print(f"  Robust Accuracy: {acc:.4f}")
    # del model_resnet_b
    # torch.cuda.empty_cache()
    
    # ------------- ResNet-C -------------
    print("\n--- ResNet-C ---")
    model_resnet_c = load_resnet(resnet_C_path, device, inputImageSize, dropOutRate, numClasses)
    acc = utils.validateD(advLoaderAPGD, model_resnet_c, device)
    results["ResNet-C"] = acc
    print(f"  Robust Accuracy: {acc:.4f}")
    del model_resnet_c
    torch.cuda.empty_cache()
    
    # # ------------- CaiT-B -------------
    # print("\n--- CaiT-B ---")
    # model_cait_b = load_cait(cait_B_path, device)
    # acc = utils.validateD(advLoaderAPGD, model_cait_b, device)
    # results["CaiT-B"] = acc
    # print(f"  Robust Accuracy: {acc:.4f}")
    # del model_cait_b
    # torch.cuda.empty_cache()
    
    # # ------------- CaiT-C -------------
    # print("\n--- CaiT-C ---")
    # model_cait_c = load_cait(cait_C_path, device)
    # acc = utils.validateD(advLoaderAPGD, model_cait_c, device)
    # results["CaiT-C"] = acc
    # print(f"  Robust Accuracy: {acc:.4f}")
    # del model_cait_c
    # torch.cuda.empty_cache()
    
    # # ------------- SVM-B -------------
    # print("\n--- SVM-B ---")
    # model_svm_b = load_svm(svm_B_base, svm_B_multi, device)
    # acc = utils.validateD(advLoaderAPGD, model_svm_b, device)
    # results["SVM-B"] = acc
    # print(f"  Robust Accuracy: {acc:.4f}")
    # del model_svm_b
    # torch.cuda.empty_cache()
    
    # # ------------- SVM-C -------------
    # print("\n--- SVM-C ---")
    # model_svm_c = load_svm(svm_C_base, svm_C_multi, device)
    # acc = utils.validateD(advLoaderAPGD, model_svm_c, device)
    # results["SVM-C"] = acc
    # print(f"  Robust Accuracy: {acc:.4f}")
    # del model_svm_c
    # torch.cuda.empty_cache()
    
    # # ===================== SUMMARY TABLE =====================
    # print(f"\n{'=' * 70}")
    # print(f"SUMMARY: Adversarial Transferability (Source: VGG-B, eps={int(epsilonMax * 255)}/255)")
    # print(f"{'=' * 70}")
    # print(f"{'Model':<15} {'Robust Acc':<15} {'Attack Success Rate':<20}")
    # print("-" * 50)
    
    # for model_name, acc in results.items():
    #     attack_success = 1 - acc
    #     marker = " (Source)" if model_name == "VGG-B" else ""
    #     print(f"{model_name + marker:<15} {acc:<15.4f} {attack_success:<20.4f}")
    
    # print("-" * 50)
    # print(f"\nNOTE: Lower Robust Accuracy = Higher Transferability")


if __name__ == '__main__':
    main()