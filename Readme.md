## TO DO

1. Have VGG-16 (synthetic) be the synthetic model, run the attack with oracle on all combined models
Also try the experiment with Carlini as the synethic model
Also try eps=255/255 for this experiment




## Results

Voters To Do

#########

Synthetic Model: CarliniNetwork
Oracle: ResNet-C
AGGD: 255/255

Total samples in valLoader: 7137
ValLoader Accuracy on Synthetic Model: 0.25360795852599133
Max (cleanLoader - advLoaderAPGD): 0.9999000430107117
Robust Accuracy APGD-DLR on synthetic Model : 0.5
Robust Accuracy APGD-DLR on Oracle 1.0
Queries used: 32000

Note: correctLoader is 1000 correctly classified samples by oracle

########

Synthetic Model: CarliniNetwork
Oracle: ResNet-C
AGGD: 8/255

Total samples in valLoader: 7137
ValLoader Accuracy on Synthetic Model: 0.25360795852599133
Max (cleanLoader - advLoaderAPGD): 0.0
Robust Accuracy APGD-DLR on synthetic Model : 0.5
Robust Accuracy APGD-DLR on Oracle 1.0
Queries used: 32000

######### 

Synthetic Model: VGG16
Oracle: ResNet-C
AGGD: 255/255

Total samples in valLoader: 7137
ValLoader Accuracy on Synthetic Model: 0.25360795852599133
Clean Accuracy on cleanLoader on Synthetic Model for samples to be attack: 0.5
Max (cleanLoader - advLoaderAPGD): 0.9999000430107117
Robust Accuracy APGD-DLR on synthetic Model : 0.5
Robust Accuracy APGD-DLR on Oracle 0.5
Queries used: 32000


#############

Synthetic Model: VGG16
Oracle: ResNet-C
AGGD: 8/255

Total samples in valLoader: 7137
ValLoader Accuracy on Synthetic Model: 0.25360795852599133
Clean Accuracy on cleanLoader on Synthetic Model for samples to be attack: 0.5
Max (cleanLoader - advLoaderAPGD): 0.03137257695198059
Robust Accuracy APGD-DLR on synthetic Model : 0.5
Robust Accuracy APGD-DLR on Oracle 1.0
Queries used: 32000


############# 

Synthetic Model: ResNet
Oracle: ResNet-C
AGGD: 8/255

ValLoader Accuracy on Synthetic Model: 0.25360795852599133
Clean Accuracy on cleanLoader on Synthetic Model for samples to be attack: 0.5
Max (cleanLoader - advLoaderAPGD): 0.03137257695198059
Robust Accuracy APGD-DLR on synthetic Model : 0.0
Robust Accuracy APGD-DLR on Oracle 1.0
Queries used: 32000





#################



###  Writing
Datsets, Models
Black Box Attacks section (Adversarial Threat Model)


Game Theoretic Mixed Experts












## BarrierZoneTrainer.py

AttackMethods.AdaptiveAttackBARZ8(modelDir)

## AttackMethods.py

def AdaptiveAttackBARZ8(modelDir):
Create a synthetic model()
syntheticModel = NetworkConstructorsAdaptive.CarliniNetwork(imgSize, numClasses)

Do the attack
AttackWrappersAdaptiveBlackBox.AdaptiveAttack(saveTag, device, oracle, syntheticModel, numIterations, epochsPerIteration, epsForAug, learningRate, optimizerName, dataLoaderForTraining, cleanLoader, valLoader, numClasses, epsForAttacks, clipMin, clipMax)


## AttackWrappersAdaptiveBlackBox.py

------ def AdaptiveAttack ------

Train Synthetic Model()
-> Label data using oracle | trainDataLoader = LabelDataUsingOracle(oracle, dataLoader, numClasses)
-> Setup Training Parameter
-> Setup Giant Datalooader ( It saves oracle labeled data)
-> Initial training (TrainingStep(device, syntheticModel, giantDataLoader, epochsPerIteration, 
                 criterion, optimizer))

                
-> Data Augmentation and Training | 
-> synthetic data are generated from the FGSM white box attack and then passing it to oracle for labeling
-> Adding sythetic data to giantloader
-> Training on giantloader with orginal + adversial
-> Complete training process

 
-> Do the white box attack
advLoaderMIM = AttackWrappersWhiteBoxP.MIMNativePytorch
-> Generate adversarial samples using MIM attack on Synthetic model

-> Evaluate attack success on real oracle
-> Save results to text file




