import torch
from torch.utils.data import TensorDataset, DataLoader

import APGDOriginal
import AttackWrappersWhiteBoxP
from DataLoaderGiant import DataLoaderGiant
import utils

from datetime import date
import os 
global queryCounter

def AdaptiveAttack(saveTag, device, oracle, syntheticModel, numClasses, training_config, attack_config):
    
    # Unpack training config
    numIterations = training_config["numIterations"]
    epochsPerIteration = training_config["epochsPerIteration"]
    epsForAug = training_config["epsForAug"]
    learningRate = training_config["learningRate"]
    optimizerName = training_config["optimizerName"]
    dataLoaderForTraining = training_config["dataLoaderForTraining"]
    valLoader = training_config["valLoader"]
    
    # Unpack attack config
    numAttackSamples = attack_config["numAttackSamples"]
    epsForAttacks = attack_config["epsForAttacks"]
    clipMin = attack_config["clipMin"]
    clipMax = attack_config["clipMax"]
    etaStart = attack_config["etaStart"]
    numSteps = attack_config["numSteps"]
    
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

    # Define Query Counter
    global queryCounter
    queryCounter = 0
    
    # Train Synthetic Model 
    TrainSyntheticModel("./", device, oracle, syntheticModel, numIterations, epochsPerIteration, epsForAug, learningRate, optimizerName, dataLoaderForTraining, numClasses, clipMin, clipMax)
    torch.save(syntheticModel, "./SyntheticModel")

    valAcc = utils.validateD(valLoader, syntheticModel, device)
    print("ValLoader Accuracy on Synthetic Model:", valAcc)  

    valAcc = utils.validateD(valLoader, oracle, device)
    print("ValLoader Accuracy on Oracle Model:", valAcc)  
    print("\n" + "-"*60)

    # oracle_correctLoader = utils.GetCorrectlyIdentifiedSamplesBalanced(oracle, numAttackSamples, valLoader, numClasses)
    correctLoader = utils.GetCorrectlyIdentifiedSamplesBalanced(syntheticModel, numAttackSamples, valLoader, numClasses)
    
    # Do the Attack | APGD DLR Attack
    advLoaderAPGD = APGDOriginal.DLR_AutoAttackPytorchMatGPUWrapper(device, correctLoader, syntheticModel, epsForAttacks, etaStart, numSteps, clipMin, clipMax)

    # Extract tensors from both dataloaders
    xClean, _ = utils.DataLoaderToTensor(correctLoader)
    xAdv, _ = utils.DataLoaderToTensor(advLoaderAPGD)
    
    # Compute max difference (L∞ norm)
    diff = torch.max(torch.abs(xClean - xAdv))
    print(f"Max (correctLoader - advLoaderAPGD): {diff}")

    advAcc = utils.validateD(advLoaderAPGD, syntheticModel, device)
    print("Sythetic Model Adversarial Acc on Synthetic adversarial loader:", advAcc)

    utils.calculateClasswiseAccuracy(advLoaderAPGD, syntheticModel, device, numClasses)

    # #Check the accuracy of the model on the adversarial examples 
    advAcc_oracle_syn = utils.validateD(advLoaderAPGD, oracle, device)
    print("Oracle Adversarial Acc on Synthetic adversarial loader:", advAcc_oracle_syn)

    utils.calculateClasswiseAccuracy(advLoaderAPGD, oracle, device, numClasses)

    print("Queries used:", queryCounter)

    # # Save adversarial samples
    # torch.save(advLoaderAPGD, "./AdvLoaderAPGD")
    # torch.cuda.empty_cache()
    
    # print("Queries used:", queryCounter)
    # # Write the results to text file 
    # resultsTextFile.write("---- ValLoader Accuracy on Synthetic Model:" + str(valAcc) + "\n")
    # resultsTextFile.write("---- Robust Accuracy APGD-DLR on synthetic Model:" + str(robustAccSynthetic) + "\n")
    # resultsTextFile.write("---- Robust Accuracy APGD-DLR on Oracle:" + str(robustAccAPGD) + "\n")
    # resultsTextFile.write("---- Queries used:" + str(queryCounter) + "\n")
    # resultsTextFile.close()  # Close the results file at the end 
    # os.chdir("..")  # Move up one directory to return to original directory 

def TrainSyntheticModel(saveDir, device, oracle, syntheticModel, numIterations, epochsPerIteration, epsForAug, learningRate, optimizerName, trainDataLoader, numClasses, clipMin, clipMax):
    # First re-label the training data according to the oracle 
    trainDataLoader = LabelDataUsingOracle(oracle, trainDataLoader, device)
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
    giantDataLoader = DataLoaderGiant(homeDir, trainDataLoader.batch_size)
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
            syntheticDataLoader = LabelDataUsingOracle(oracle, syntheticDataLoaderUnlabeled, device)
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

def LabelDataUsingOracle(oracle, dataLoader, device):
    oracle.eval()
    numSamples = len(dataLoader.dataset)

    #Update the query Counter
    global queryCounter
    queryCounter = queryCounter + numSamples
    
    # Collect all data, original labels, and predictions
    all_inputs = []
    all_original_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, original_labels in dataLoader:
            inputs = inputs.to(device)
            outputs = oracle(inputs)
            
            # Get hard labels (argmax of predictions)
            predictions = outputs.argmax(dim=1)
            
            all_inputs.append(inputs.cpu())
            all_original_labels.append(original_labels.cpu())
            all_predictions.append(predictions.cpu())
    
    # Concatenate all batches
    xData = torch.cat(all_inputs, dim=0)
    yOriginal = torch.cat(all_original_labels, dim=0)
    yLabels = torch.cat(all_predictions, dim=0)
    
    # -------- LABEL COMPARISON STATISTICS --------
    print("\n" + "-"*60)
    print("LABEL COMPARISON STATISTICS")
    
    # Overall statistics
    same_labels = (yOriginal == yLabels).sum().item()
    different_labels = (yOriginal != yLabels).sum().item()
    total_samples = len(yLabels)
    
    print(f"Total Samples: {total_samples}")
    print(f"Labels Unchanged: {same_labels} ({100*same_labels/total_samples:.2f}%)")
    print(f"Labels Changed:   {different_labels} ({100*different_labels/total_samples:.2f}%)")
    print("-"*60)
    
    # Create new DataLoader with oracle labels
    labeledDataset = TensorDataset(xData, yLabels)
    dataLoaderLabeled = DataLoader(
        labeledDataset, 
        batch_size=dataLoader.batch_size, 
        shuffle=True
    )
    
    return dataLoaderLabeled

