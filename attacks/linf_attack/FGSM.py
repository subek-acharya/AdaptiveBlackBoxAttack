#Attack wrappers class for FGSM and MIM (no extra library implementation) to be used in conjunction with 
#the adaptive black-box attack 
import torch 
import utils as DMP
import torchvision

#Native (no attack library) implementation of the FGSM attack in Pytorch 
def FGSMNativePytorch(device, dataLoader, model, epsilonMax, clipMin, clipMax, targeted):
    model.eval() #Change model to evaluation mode for the attack 
    #Generate variables for storing the adversarial examples 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    #Go through each sample 
    tracker = 0
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        #print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device 
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
        xDataTemp.requires_grad = True
        # Forward pass the data through the model
        output = model(xDataTemp)
        # Calculate the loss
        loss = torch.nn.CrossEntropyLoss()
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = loss(output, yData).to(device)
        cost.backward()
        # Collect datagrad
        #xDataGrad = xDataTemp.grad.data
        ###Here we actual compute the adversarial sample 
        # Collect the element-wise sign of the data gradient
        signDataGrad = xDataTemp.grad.data.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        #print("xData:", xData.is_cuda)
        #print("SignGrad:", signDataGrad.is_cuda)
        if targeted == True:
            perturbedImage = xData - epsilonMax*signDataGrad.cpu().detach() #Go negative of gradient
        else:
            perturbedImage = xData + epsilonMax*signDataGrad.cpu().detach()
        # Adding clipping to maintain the range
        perturbedImage = torch.clamp(perturbedImage, clipMin, clipMax)
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = perturbedImage[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
        #Not sure if we need this but do some memory clean up 
        del xDataTemp
        del signDataGrad
        torch.cuda.empty_cache()
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader