import torch

from model_architecture import ResNet, cait, VGG, MultiOutputSVM, CarliniNetwork
import AttackWrappersAdaptiveBlackBox
from ModelFactory import ModelFactory
import utils

def AdaptiveAttack():
    # --------------- MODEL PATHS ------------------
    resnet_C_path = "./checkpoint/ModelResNet20-VotingCombined-v2-Grayscale-Run1.th"
    cait_C_path = "./checkpoint/ModelCaiT-trCombined-v2-valCombined-v2-Grayscale-Run1.th"
    vgg_C_path = "./checkpoint/ModelVgg16-C2.th"
    svm_C_base = "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/base_pytorch_svm_combined_v2.pth"
    svm_C_multi = "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/multi_output_svm_combined_v2.pth"

    # ----------------- PARAMETERS --------------------
    batchSize = 64
    numClasses = 2
    inputImageSize = [1, 1, 40, 50]
    dropOutRate = 0.0
    imgH, imgW = 40, 50
    numTrainingSamples = 2000
    
    saveTag ="Adaptive Attack"
    device = torch.device("cuda")

    # ------------------ ORACLE MODELS ------------------------
    oracle = ModelFactory().get_model('resnet', resnet_C_path)
    # oracle = ModelFactory().get_model('cait', cait_C_path)
    # oracle = ModelFactory().get_model('vgg', vgg_C_path)
    # oracle = ModelFactory().get_model('svm', [svm_C_base, svm_C_multi])# Synthetic Model

    # ------------------ SYNTHETIC (UNTRAINED) MODELS ------------------------
    # syntheticModel = ModelFactory().get_model('resnet')
    # syntheticModel = ModelFactory().get_model('cait')
    # syntheticModel = ModelFactory().get_model('vgg')
    # syntheticModel = ModelFactory().get_model('svm')
    syntheticModel = ModelFactory().get_model('carlini')

    # -------------- TRAINING & VALIDATION DATASET ------
    trainLoader = utils.GetVoterTrainingBalanced(batchSize, numTrainingSamples, numClasses)
    valLoader = utils.GetVoterValidation(batchSize)
    
    # -------------- TRAINING CONFIG ------------------
    training_config = {
        "batchSize": batchSize,
        "numIterations": 4,
        "epochsPerIteration": 10,
        "epsForAug": 0.1,
        "learningRate": 0.0001,
        "numTrainingSamples": numTrainingSamples,
        "dataLoaderForTraining": trainLoader,
        "valLoader": valLoader,
        "optimizerName": "adam",
    }
    
    # -------------- ATTACK CONFIG (APGD DLR Attack) --------------------
    attack_config = {
        "numAttackSamples": 1000,
        "epsForAttacks": 16/255,
        "clipMin": 0.0,
        "clipMax": 1.0,
        "etaStart": 2 * (16/255),
        "numSteps": 500,
    }
        
    # Run the attack
    AttackWrappersAdaptiveBlackBox.AdaptiveAttack(
        saveTag=saveTag,
        device=device,
        oracle=oracle,
        syntheticModel=syntheticModel,
        numClasses=numClasses,
        training_config=training_config,
        attack_config=attack_config
    )
    
def main():
    # Run the adaptive black-box attack
    AdaptiveAttack()

if __name__ == '__main__':
    main()