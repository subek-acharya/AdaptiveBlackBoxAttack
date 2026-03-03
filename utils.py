import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# ------------ Voters specific utils -----------------

def GetVoterValidation(batchSize):
    valData = torch.load("./data/kaleel_final_dataset_val_OnlyBubbles_Grayscale.pth", weights_only=False)
    valImages = valData["data"].float()
    valLabels = valData["binary_labels"].long()
    
    valDataset = TensorDataset(valImages, valLabels)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False)
    return valLoader

def GetVoterTraining(batchSize):
    trainData = torch.load("./data/kaleel_final_dataset_train_OnlyBubbles_Grayscale.pth", weights_only=False)
    trainImages = trainData["data"].float()
    trainLabels = trainData["binary_labels"].long()
    
    trainDataset = TensorDataset(trainImages, trainLabels)
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=False)
    return trainLoader


def GetVoterTrainingBalanced(batchSize, totalSamples, numClasses):
    # Get all training data (shuffled) with same batchSize
    fullTrainLoader = GetVoterTraining(batchSize=batchSize)
    
    # Collect all shuffled data from batches
    allImages = []
    allLabels = []
    for images, labels in fullTrainLoader:
        allImages.append(images)
        allLabels.append(labels)
    
    trainImages = torch.cat(allImages, dim=0)
    trainLabels = torch.cat(allLabels, dim=0)
    
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
        
        if classCount[label] < samplesPerClass:
            balancedImages[currentIndex] = trainImages[i]
            balancedLabels[currentIndex] = label
            classCount[label] += 1
            currentIndex += 1
        
        if currentIndex >= totalSamples:
            break
    
    # Verify we got enough samples
    for c in range(numClasses):
        if classCount[c] != samplesPerClass:
            raise ValueError(f"Not enough samples for class {c}. Got {int(classCount[c])}, needed {samplesPerClass}")
    
    print(f"Balanced training data: {totalSamples} samples ({samplesPerClass} per class)")
    
    # Create dataloader
    balancedDataset = TensorDataset(balancedImages, balancedLabels.long())
    balancedLoader = DataLoader(balancedDataset, batch_size=batchSize, shuffle=False)
    
    return balancedLoader

# Calculate and print class-wise accuracy for a given model and dataloader
def calculateClasswiseAccuracy(dataLoader, model, device, numClasses):
    model.eval()
    
    # Initialize counters for each class
    correct_per_class = {i: 0 for i in range(numClasses)}
    total_per_class = {i: 0 for i in range(numClasses)}
    
    with torch.no_grad():
        for inputs, labels in dataLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Count correct predictions per class
            for label, pred in zip(labels, predicted):
                label_idx = label.item()
                total_per_class[label_idx] += 1
                if label_idx == pred.item():
                    correct_per_class[label_idx] += 1
    
    # Calculate accuracies
    classwise_acc = {}
    print(f"\n{'='*50}")
    print(f"Class-wise Accuracy")
    print(f"{'='*50}")
    print(f"{'Class':<10} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print(f"{'-'*50}")
    
    total_correct = 0
    total_samples = 0
    
    for cls in range(numClasses):
        if total_per_class[cls] > 0:
            acc = correct_per_class[cls] / total_per_class[cls]
        else:
            acc = 0.0
        classwise_acc[cls] = acc
        total_correct += correct_per_class[cls]
        total_samples += total_per_class[cls]
        
        print(f"{cls:<10} {correct_per_class[cls]:<10} {total_per_class[cls]:<10} {acc:.4f}")
    
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"{'-'*50}")
    print(f"{'Overall':<10} {total_correct:<10} {total_samples:<10} {overall_acc:.4f}")
    print(f"{'='*50}\n")
    
    return overall_acc, classwise_acc
# --------------------------------------------

#Convert a dataloader into x and y tensors 
def DataLoaderToTensor(dataLoader):
    #First check how many samples in the dataset
    numSamples = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    #xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    xData = torch.zeros((numSamples,) + sampleShape) #Make it generic shape for non-image datasets
    yData = torch.zeros(numSamples)
    #Go through and process the data in batches 
    for i, (input, target) in enumerate(dataLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xData[sampleIndex] = input[batchIndex]
            yData[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    return xData, yData

#Convert a X and Y tensors into a dataloader
#Does not put any transforms with the data  
def TensorToDataLoader(xData, yData, transforms= None, batchSize=None, randomizer = None):
    if batchSize is None: #If no batch size put all the data through 
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None: #No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, shuffle=False)
    else: #randomizer needed 
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader

def TensorToNumpy(x_tensor, y_tensor):
    x_numpy = x_tensor.cpu().numpy()
    y_numpy = y_tensor.cpu().numpy().astype(np.int64)
    return x_numpy, y_numpy

def NumpyToTensor(x_numpy, y_numpy):
    x_tensor = torch.from_numpy(x_numpy).float()
    y_tensor = torch.from_numpy(y_numpy).long()  # long is int64
    return x_tensor, y_tensor


# Find the actual min and max pixel values in the dataset
def GetDataBounds(dataLoader, device):
    minVal = float('inf')
    maxVal = float('-inf')
    
    for xData, _ in dataLoader:
        xData = xData.to(device)
        batchMin = xData.min().item()
        batchMax = xData.max().item()
        
        if batchMin < minVal:
            minVal = batchMin
        if batchMax > maxVal:
            maxVal = batchMax
    
    return minVal, maxVal

#Validate using a dataloader 
def validateD(valLoader, model, device=None):
    #switch to evaluate mode
    model.eval()
    acc = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None: #assume cuda
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    acc = acc +1
    acc = acc / float(len(valLoader.dataset))
    return acc

def GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, dataLoader, numClasses, device=None):
    model.eval()
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = predictD(dataLoader, numClasses, model, device)
    a = 0
    for i in range(0, xData.shape[0]): #Go through every sample 
        a = a + 1
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

#Get the output shape from the dataloader
def GetOutputShape(dataLoader):
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape

#Replicate TF's predict method behavior 
def predictD(dataLoader, numClasses, model, device=None):
    numSamples = len(dataLoader.dataset)
    yPred = torch.zeros(numSamples, numClasses)
    #switch to evaluate mode
    model.eval()
    indexer = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None:
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            for j in range(0, sampleSize):
                yPred[indexer] = output[j]
                indexer = indexer + 1 #update the indexer regardless 
    return yPred

#Class to help with converting between dataloader and pytorch tensor 
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None: #No transform so return the data directly
            return (self.x[index], self.y[index])
        else: #Transform so apply it to the data before returning 
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)
