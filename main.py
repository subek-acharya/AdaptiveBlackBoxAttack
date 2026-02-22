import torch
from torch.utils.data import TensorDataset, DataLoader

from model_architecture import VGG, ResNet
import utils
import NetworkConstructorsAdaptive
import AttackWrappersAdaptiveBlackBox


def LoadVGG(modelDir, device, imgH, imgW, numClasses):
    model = VGG.VGG("VGG16", imgH, imgW, numClasses).to(device)
    raw = torch.load(modelDir, map_location="cpu")
    state = raw.get("state_dict", raw)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def load_resnet(modelDir, device, inputImageSize, dropOutRate, numClasses):
    """Load ResNet model"""
    model = ResNet.resnet20(inputImageSize, dropOutRate, numClasses).to(device)
    checkpoint = torch.load(modelDir, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def GetVoterTrainingBalanced(batchSize, totalSamples, numClasses):
    """
    Load balanced voter training dataset
    
    Args:
        batchSize: Batch size for dataloader
        totalSamples: Total number of samples to get (e.g., 2000)
        numClasses: Number of classes (e.g., 2)
    
    Returns:
        DataLoader with balanced samples (totalSamples/numClasses per class)
    """
    trainData = torch.load("./data/kaleel_final_dataset_train_OnlyBubbles_Grayscale.pth", weights_only=False)
    trainImages = trainData["data"].float()
    trainLabels = trainData["binary_labels"].long()
    
    # Calculate samples per class
    samplesPerClass = totalSamples // numClasses
    
    # Get shape of images
    imgShape = trainImages[0].shape
    
    # Initialize tensors for balanced data
    balancedImages = torch.zeros(totalSamples, imgShape[0], imgShape[1], imgShape[2])
    balancedLabels = torch.zeros(totalSamples)
    
    # Track how many samples we've collected per class
    classCount = torch.zeros(numClasses)
    
    # Collect balanced samples
    currentIndex = 0
    for i in range(len(trainLabels)):
        label = int(trainLabels[i])
        
        # Check if we still need samples from this class
        if classCount[label] < samplesPerClass:
            balancedImages[currentIndex] = trainImages[i]
            balancedLabels[currentIndex] = label
            classCount[label] += 1
            currentIndex += 1
        
        # Check if we have enough samples
        if currentIndex >= totalSamples:
            break
    
    # Verify we got enough samples
    for c in range(numClasses):
        if classCount[c] != samplesPerClass:
            raise ValueError(f"Not enough samples for class {c}. Got {int(classCount[c])}, needed {samplesPerClass}")
    
    print(f"Balanced training data: {totalSamples} samples ({samplesPerClass} per class)")
    
    # Create dataloader
    balancedDataset = TensorDataset(balancedImages, balancedLabels.long())
    balancedLoader = DataLoader(balancedDataset, batch_size=batchSize, shuffle=True)
    
    return balancedLoader


def GetVoterValidation(batchSize):
    valData = torch.load("./data/kaleel_final_dataset_val_OnlyBubbles_Grayscale.pth", weights_only=False)
    valImages = valData["data"].float()
    valLabels = valData["binary_labels"].long()
    
    valDataset = TensorDataset(valImages, valLabels)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False)
    return valLoader


def AdaptiveAttack():

    # --------------- MODEL PATHS ------------------
    # ResNet model
    resnet_C_path = "./checkpoint/ModelResNet20-VotingCombined-v2-Grayscale-Run1.th"
    # CaiT model
    cait_C_path = "./checkpoint/ModelCaiT-trCombined-v2-valCombined-v2-Grayscale-Run1.th"
    # VGG model
    vgg_C_path = "./checkpoint/ModelVgg16-C2.th"
    # SVM model
    svm_C_base = "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/base_pytorch_svm_combined_v2.pth"
    svm_C_multi = "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/multi_output_svm_combined_v2.pth"

    # ----------------- PARAMETERS --------------------
    batchSize = 64
    numClasses = 2
    inputImageSize = [1, 1, 40, 50]
    dropOutRate = 0.0
    imgH, imgW = 40, 50
    
    saveTag ="Adaptive Attack"
    device = torch.device("cuda")
    
    # -------------- Load the oracle -----------------
    
    # Load (VGG-C)
    # oracle = LoadVGG(vgg_C_path, device, imgH, imgW, numClasses)
    # Load Resnet-C
    oracle = load_resnet(resnet_C_path, device, inputImageSize, dropOutRate, numClasses)
    
    # Synthetic Model Training Parameters
    batchSize = 64
    numIterations = 4
    epochsPerIteration = 10
    epsForAug = 0.1  # When generating synthetic data, this value is eps for FGSM
    learningRate = 0.0001  # Learning rate of the synthetic model
    numTrainingSamples = 2000 # Total balanced samples for training

    # Attack parameters (APGD DLR Attack)
    numAttackSamples = 1000
    epsForAttacks = 8/255
    clipMin = 0.0
    clipMax = 1.0
    etaStart = 2 * epsForAttacks
    numSteps = 500  # Number of attack iterations
    
    # Load the balanced training dataset (2000 samples, 1000 per class)
    trainLoader = GetVoterTrainingBalanced(batchSize, numTrainingSamples, numClasses)

    # Load the validation dataset
    valLoader = GetVoterValidation(batchSize)

    #Get the clean data 
    cleanLoader = utils.GetCorrectlyIdentifiedSamplesBalanced(oracle, numAttackSamples, valLoader, numClasses)
    
    # Create the synthetic model
    # ---- Carlini ------
    # syntheticModel = NetworkConstructorsAdaptive.CarliniNetwork(imgH=imgH, imgW=imgW, numChannels=1, numClasses=numClasses).to(device)
    # ---- VGG16 --------
    syntheticModel = VGG.VGG("VGG16", imgH, imgW, numClasses).to(device)
    # ----- ResNet --------
    # syntheticModel = ResNet.resnet20(inputImageSize, dropOutRate, numClasses).to(device)
    
    # Do the attack
    dataLoaderForTraining = trainLoader
    optimizerName = "adam"
    
    # Run the attack
    AttackWrappersAdaptiveBlackBox.AdaptiveAttack(saveTag, device, oracle, syntheticModel, numIterations, epochsPerIteration, epsForAug, learningRate, optimizerName, dataLoaderForTraining, valLoader, numClasses, epsForAttacks, clipMin, clipMax, etaStart, numSteps, numAttackSamples, cleanLoader)

def main():
    # Run the adaptive black-box attack
    AdaptiveAttack()

if __name__ == '__main__':
    main()