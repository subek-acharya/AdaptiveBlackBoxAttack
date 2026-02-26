import torch

from model_architecture import ResNet, cait, VGG, MultiOutputSVM, CarliniNetwork
import AttackWrappersAdaptiveBlackBox
import ModelFactory
import utils

def AdaptiveAttack():
    # ----------------- PARAMETERS --------------------
    batchSize = 64
    numClasses = 2
    inputImageSize = [1, 1, 40, 50]
    dropOutRate = 0.0
    imgH, imgW = 40, 50
    numTrainingSamples = 2000
    
    saveTag ="Adaptive Attack"
    device = torch.device("cuda")

    # ------------------ MODELS ------------------------
    # Oracle
    oracle = ModelFactory.ModelFactory().get_model('resnet', "./checkpoint/ModelResNet20-VotingCombined-v2-Grayscale-Run1.th")  # Resnet-C
    # Synthetic Model
    syntheticModel = ModelFactory.ModelFactory().get_model("vgg")  # VGG

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
        "epsForAttacks": 8/255,
        "clipMin": 0.0,
        "clipMax": 1.0,
        "etaStart": 2 * (8/255),
        "numSteps": 500,
    }
        
    # Run the attack - using **config to unpack dictionary
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