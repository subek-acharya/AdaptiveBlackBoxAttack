import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time

import utils
from model_architecture import VGG

from DataLoaderGiant import DataLoaderGiant
import AttackWrappersAdaptiveBlackBox

# Global variables
device = None
model = None
criterion = None
optimizer = None
trainLoader = None
valLoader = None
best_acc = 0

def LoadVGG(modelDir, device, imgH, imgW, numClasses):
    model = VGG.VGG("VGG16", imgH, imgW, numClasses).to(device)
    raw = torch.load(modelDir, map_location="cpu")
    state = raw.get("state_dict", raw)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def train(epoch, giantDataLoader):
    global model, criterion, optimizer, device
    
    print('\nEpoch: %d' % epoch)
    model.train()
    
    train_loss = 0
    correct = 0
    total = 0

    numDataLoaders = giantDataLoader.GetNumberOfLoaders()  # Find out how many loaders we have to iterate over
    for e in range(0, epoch):
        print("--Epoch=", e)
        # Go through all dataloaders 
        for loaderIndex in range(0, numDataLoaders):
            print("----Training on data loader=", loaderIndex)
            dataLoader = giantDataLoader.GetLoaderAtIndex(loaderIndex)
            # Go through all the samples in the loader
            for batch_idx, (inputs, targets) in enumerate(dataLoader):
                inputs, targets = inputs.to(device), targets.to(device).long()
                optimizer.zero_grad()       # Make all gradient values 0 at model parameter
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()      # Calculating gradients
                optimizer.step()     # Updating the weights
        
                train_loss += loss.item()   # total loss for this batch
                _, predicted = outputs.max(1)   # it returns two tensors value,index and we only need index
                total += targets.size(0)   # total data processes so far
                correct += predicted.eq(targets).sum().item()


def main():
    global device, model, criterion, optimizer, trainLoader, valLoader, best_acc

    # Define the GPU device we are using 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check available no of GPU/used for torch.nn.DataParallel
    print(f"Number of GPUs available: {torch.cuda.device_count()}")

    # Define Batch Size
    batchSize = 64
    numClasses = 2
    imgH, imgW = 40, 50
    
    best_acc = 0
    num_epochs = 10 
    learning_rate = 0.001

    vgg_C_path = "./checkpoint/ModelVgg16-C2.th"
    
    # Load (VGG-C)
    oracle = LoadVGG(vgg_C_path, device, imgH, imgW, numClasses)

    # Load data to dataloader
    
    # Creating the voter training dataloader
    # trainLoader = utils.GetVoterTrain(batchSize)
    trainLoader = utils.GetVoterTrainingBalanced(batchSize, 2000, 2)  
    # Creating the voter validation dataloader
    valLoader = utils.GetVoterValidation(batchSize)  

    labeltrainLoader = AttackWrappersAdaptiveBlackBox.LabelDataUsingOracle(oracle, trainLoader, numClasses, device)

    # Quick check in main.py
    xOrig, yOrig = utils.DataLoaderToTensor(trainLoader)
    xLabel, yLabel = utils.DataLoaderToTensor(labeltrainLoader)
    
    print(f"Images same: {torch.allclose(xOrig, xLabel)}")
    print(f"Labels same: {(yOrig == yLabel).all().item()}")
    print(f"Different labels: {(yOrig != yLabel).sum().item()}/{len(yOrig)}")

    giantDataLoader = DataLoaderGiant(labeltrainLoader.batch_size)
    giantDataLoader.AddLoader("OriginalLoader", labeltrainLoader)
        
    # Test the loaders
    print("==> Testing data loaders...")
    for images, labels in trainLoader:
        print("total data:", len(trainLoader.dataset))
        print(f"Train batch shape: {images.shape}")
        print(f"Train labels shape: {labels.shape}")
        print(f"Train data range: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    for images, labels in valLoader:
        print(f"Validation batch shape: {images.shape}")
        print(f"Validation labels shape: {labels.shape}")
        print(f"Validation data range: [{images.min():.3f}, {images.max():.3f}]")
        break

    # Define model
    model = VGG.VGG("VGG16", imgH, imgW, numClasses).to(device)

    # model = torch.nn.DataParallel(model)
    # cudnn.benchmark = True

    # Defining Loss Function
    criterion = nn.CrossEntropyLoss()

    # Defining Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    # Start tracking total training time
    print("\n" + "="*60)
    print("STARTING TRAINING...")
    print("="*60)
    training_start_time = time.time()

    # Run training (no test step during training)
    # for epoch in range(num_epochs):
    train(num_epochs, giantDataLoader)

    # Calculate total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    # ============================================================
    # VALIDATION AFTER TRAINING COMPLETED
    # ============================================================
    print("\n" + "="*60)
    print("RUNNING VALIDATION...")
    print("="*60)

    model.eval()
    
    # Get validation accuracy using utils.validateD
    val_accuracy = utils.validateD(valLoader, model, device)
    
    print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")
    
    # Optional: Get class-wise accuracy (if you have 2 classes for binary classification)
    num_classes = 2  # Adjust based on your dataset
    overall_acc, classwise_acc = utils.calculateClasswiseAccuracy(valLoader, model, device, num_classes)

    # ============================================================
    # SAVE MODEL CHECKPOINT
    # ============================================================
    print("\n" + "="*60)
    print("SAVING MODEL...")
    print("="*60)
    
    state = {
        'model': model.state_dict(),
        'acc': val_accuracy * 100,
        'epoch': num_epochs,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './synthetic_vgg16.pth')
    print(f"Model saved to ./checkpoint/vgg16_voter.pth")

    # Display comprehensive training summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Total epochs: {num_epochs}")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Total training time: {total_training_time/60:.2f} minutes")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()