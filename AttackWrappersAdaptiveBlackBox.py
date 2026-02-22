# Pytorch version of the adaptive black-box attack 
# Modified from KM's ICCV Vision code for voter dataset
import torch
import AttackWrappersWhiteBoxP
import utils as DMP
from DataLoaderGiant import DataLoaderGiant
from datetime import date
import os 

global queryCounter  # Keep track of the numbers of queries used in the adaptive black-box attack, just for record keeping

import APGDOriginal

# Main attack method 
def AdaptiveAttack(saveTag, device, oracle, syntheticModel, numIterations, epochsPerIteration, epsForAug, learningRate, optimizerName, dataLoaderForTraining, valLoader, numClasses, epsForAttacks, clipMin, clipMax, etaStart, numSteps, numAttackSamples, cleanLoader):

    #Create place to save all files
    today = date.today()
    dateString = today.strftime("%B"+"-"+"%d"+"-"+"%Y, ") #Get the year, month, day
    experimentDateAndName = dateString + saveTag #Name of experiment with data 
    saveDir = os.path.join(os.getcwd(), experimentDateAndName)
    if not os.path.isdir(saveDir): #If not there, make the directory 
        os.makedirs(saveDir)
    #Place to save the results 
    os.chdir(saveDir)
    resultsTextFile = open(experimentDateAndName+", Results.txt","a+")
    
    # Reset the query counter 
    global queryCounter
    queryCounter = 0
    
    # First train the synthetic model 
    TrainSyntheticModel("./", device, oracle, syntheticModel, numIterations, epochsPerIteration, epsForAug, learningRate, optimizerName, dataLoaderForTraining, numClasses, clipMin, clipMax)
    torch.save(syntheticModel, "./SyntheticModel")

    # Print total number of samples in valLoader
    total_sample = len(valLoader.dataset)
    print(f"Total samples in valLoader: {total_sample}")

    valAcc = DMP.validateD(valLoader, syntheticModel, device)
    print("ValLoader Accuracy on Synthetic Model:", valAcc)   

    # Classwise Accuracy for ValLoader on Synthetic Model
    print("ClasswiseAccuracy ValLoader Accuracy on Synthetic Model")
    DMP.calculateClasswiseAccuracy(valLoader, syntheticModel, device, numClasses)

    # # Get the clean data (correctly classified balanced samples)
    # cleanLoader = DMP.GetCorrectlyIdentifiedSamplesBalanced(syntheticModel, numAttackSamples, valLoader, numClasses)
    cleanAcc = DMP.validateD(cleanLoader, syntheticModel, device)
    print("Clean Accuracy on cleanLoader on Synthetic Model for samples to be attack:", cleanAcc)

    # Do the Attack | APGD DLR Attack
    advLoaderAPGD = APGDOriginal.DLR_AutoAttackPytorchMatGPUWrapper(device, cleanLoader, syntheticModel, epsForAttacks, etaStart, numSteps, clipMin, clipMax)

    # Save adversarial samples
    torch.save(advLoaderAPGD, "./AdvLoaderAPGD")
    torch.cuda.empty_cache()

    # Extract tensors from both dataloaders
    xClean, yClean = DMP.DataLoaderToTensor(cleanLoader)
    xAdv, yAdv = DMP.DataLoaderToTensor(advLoaderAPGD)
    
    # Compute max difference (L∞ norm)
    diff = torch.max(torch.abs(xClean - xAdv))
    print(f"Max (cleanLoader - advLoaderAPGD): {diff}")

    # ADDED: Evaluate adversarial robustness on synthetic model
    robustAccSynthetic = DMP.validateD(advLoaderAPGD, syntheticModel, device)
    print("Robust Accuracy APGD-DLR on synthetic Model :", robustAccSynthetic)

    # Classwise Accuracy for advLoaderAPGD on Synthetic Model
    print("ClasswiseAccuracy advLoaderAPGD Accuracy on Synthetic Model")
    DMP.calculateClasswiseAccuracy(advLoaderAPGD, syntheticModel, device, numClasses)
    
    # Evaluate on oracle
    robustAccAPGD = DMP.validateD(advLoaderAPGD, oracle, device)
    print("Robust Accuracy APGD-DLR on Oracle", robustAccAPGD)

    # Classwise Accuracy for advLoaderAPGD on Oracle
    print("ClasswiseAccuracy advLoaderAPGD Accuracy on Oracle")
    DMP.calculateClasswiseAccuracy(advLoaderAPGD, oracle, device, numClasses)
    
    print("Queries used:", queryCounter)
    # Write the results to text file 
    resultsTextFile.write("---- ValLoader Accuracy on Synthetic Model:" + str(valAcc) + "\n")
    resultsTextFile.write("---- Robust Accuracy APGD-DLR on synthetic Model:" + str(robustAccSynthetic) + "\n")
    resultsTextFile.write("---- Robust Accuracy APGD-DLR on Oracle:" + str(robustAccAPGD) + "\n")
    resultsTextFile.write("---- Queries used:" + str(queryCounter) + "\n")
    resultsTextFile.close()  # Close the results file at the end 
    os.chdir("..")  # Move up one directory to return to original directory 


# Method to label the data using the oracle 
# In original implementation, Adversarial filtering is also used, whereas in this case, no adversarial filtering used since we have two class and model does not detect adversarial class
def LabelDataUsingOracle(oracle, dataLoader, numClasses, device):
    global queryCounter
    numSamples = len(dataLoader.dataset)
    # Update the query counter 
    queryCounter = queryCounter + numSamples 
    # Do the prediction 
    yPredOracle = DMP.predictD(dataLoader, numClasses, oracle, device) 
    # Convert to hard labels
    yHardOracle = torch.zeros(numSamples)
    for i in range(0, numSamples):
        yHardOracle[i] = int(yPredOracle[i].argmax(axis=0))
    # Get the x data from the dataloader
    xData, yWrong = DMP.DataLoaderToTensor(dataLoader)  # Note we don't care about yWrong, just don't use it
    # Put the tensors in a dataloader and return 
    dataLoaderLabeled = DMP.TensorToDataLoader(xData, yHardOracle, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)
    return dataLoaderLabeled 


def TrainSyntheticModel(saveDir, device, oracle, syntheticModel, numIterations, epochsPerIteration, epsForAug, learningRate, optimizerName, dataLoader, numClasses, clipMin, clipMax):
    # First re-label the training data according to the oracle 
    trainDataLoader = LabelDataUsingOracle(oracle, dataLoader, numClasses, device)
    # Setup the training parameters 
    criterion = torch.nn.CrossEntropyLoss()
    # Check what optimizer to use
    if optimizerName == "adam":
        optimizer = torch.optim.Adam(syntheticModel.parameters(), lr=learningRate)
    elif optimizerName == "sgd":
        optimizer = torch.optim.SGD(syntheticModel.parameters(), lr=learningRate, momentum=0.9, weight_decay=0)
    else:
        raise ValueError("Optimizer name not recognized.")
    # Setup the giant data loader
    homeDir = "./"
    giantDataLoader = DataLoaderGiant(homeDir, dataLoader.batch_size)
    giantDataLoader.AddLoader("OriginalLoader", trainDataLoader)
    # Do one round of training with the currently labeled training data 
    TrainingStep(device, syntheticModel, giantDataLoader, epochsPerIteration, criterion, optimizer)
    # Data augmentation and training steps 
    for i in range(0, numIterations):
        print("Running synthetic model training iteration =", i)
        # Create the synthetic data using FGSM and the synthetic model 
        numDataLoaders = giantDataLoader.GetNumberOfLoaders()  # Find out how many loaders we have to iterate over
        # Go through and generate adversarial examples for each dataloader
        print("=Step 0: Generating data loaders...")
        for j in range(0, numDataLoaders):
            print("--Generating data loader=", j)
            currentLoader = giantDataLoader.GetLoaderAtIndex(j)
            syntheticDataLoaderUnlabeled = AttackWrappersWhiteBoxP.FGSMNativePytorch(device, currentLoader, syntheticModel, epsForAug, clipMin, clipMax, targeted=False)
            # Memory clean up 
            del currentLoader
            # Label the synthetic data using the oracle 
            syntheticDataLoader = LabelDataUsingOracle(oracle, syntheticDataLoaderUnlabeled, numClasses, device)
            # Memory clean up
            del syntheticDataLoaderUnlabeled
            giantDataLoader.AddLoader("DataLoader,iteration=" + str(i) + "batch=" + str(j), syntheticDataLoader)          
        # Combine the new synthetic data loader and the original data loader
        print("=Step 1: Training the synthetic model...")
        # Train on the new data 
        TrainingStep(device, syntheticModel, giantDataLoader, epochsPerIteration, criterion, optimizer)


# Try to match Keras "fit" function as closely as possible 
def TrainingStep(device, model, giantDataLoader, numEpochs, criterion, optimizer):
    # Switch into training mode 
    model.train()
    numDataLoaders = giantDataLoader.GetNumberOfLoaders()  # Find out how many loaders we have to iterate over
    for e in range(0, numEpochs):
        print("--Epoch=", e)
        # Go through all dataloaders 
        for loaderIndex in range(0, numDataLoaders):
            print("----Training on data loader=", loaderIndex)
            dataLoader = giantDataLoader.GetLoaderAtIndex(loaderIndex)
            # Go through all the samples in the loader
            for i, (input, target) in enumerate(dataLoader):
                targetVar = target.to(device).long()
                inputVar = input.to(device)
                # Compute output
                output = model(inputVar)
                loss = criterion(output, targetVar)
                # Compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        del dataLoader
        del inputVar
        del targetVar
        torch.cuda.empty_cache()