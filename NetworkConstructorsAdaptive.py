# Network constructors for the adaptive black-box attack 
# Modified for grayscale voter data (1 channel, 40x50)
import torch.nn
import torch.nn.functional as F


class CarliniNetwork(torch.nn.Module):
    def __init__(self, imgH=40, imgW=50, numChannels=1, numClasses=2):
        """
        Carlini Network adapted for grayscale voter data
        """
        super(CarliniNetwork, self).__init__()
        
        # Parameters for the network 
        params = [64, 64, 128, 128, 256, 256]
        
        # Store image dimensions
        self.imgH = imgH
        self.imgW = imgW
        self.numChannels = numChannels
        
        # Create the layers
        # Conv2D(params[0], (3, 3), input_shape=inputShape) + Activation('relu')
        self.conv0 = torch.nn.Conv2d(in_channels=numChannels, out_channels=params[0], kernel_size=(3,3), stride=1)
        
        # Conv2D(params[1], (3, 3)) + Activation('relu')
        self.conv1 = torch.nn.Conv2d(in_channels=params[0], out_channels=params[1], kernel_size=(3,3), stride=1)
        
        # MaxPooling2D(pool_size=(2, 2))
        self.mp0 = torch.nn.MaxPool2d(kernel_size=(2,2))
        
        # Conv2D(params[2], (3, 3)) + Activation('relu')
        self.conv2 = torch.nn.Conv2d(in_channels=params[1], out_channels=params[2], kernel_size=(3,3), stride=1)
        
        # Conv2D(params[3], (3, 3)) + Activation('relu')
        self.conv3 = torch.nn.Conv2d(in_channels=params[2], out_channels=params[3], kernel_size=(3,3), stride=1)
        
        # MaxPooling2D(pool_size=(2, 2))
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2,2))
        
        # Compute flatten size dynamically
        testInput = torch.zeros((1, numChannels, imgH, imgW))
        outputShape = self.figureOutFlattenShape(testInput)
        
        # Dense(params[4]) + Activation('relu')
        self.forward0 = torch.nn.Linear(in_features=outputShape[1], out_features=params[4])
        
        # Dropout(0.5)
        self.drop0 = torch.nn.Dropout(0.5)
        
        # Dense(params[5]) + Activation('relu')
        self.forward1 = torch.nn.Linear(in_features=params[4], out_features=params[5])
        
        # Dense(numClasses) + Activation('softmax')
        self.forward2 = torch.nn.Linear(in_features=params[5], out_features=numClasses)

    def forward(self, x):
        out = F.relu(self.conv0(x))
        out = F.relu(self.conv1(out))
        out = self.mp0(out)
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.mp1(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = F.relu(self.forward0(out))
        out = self.drop0(out)
        out = F.relu(self.forward1(out))
        out = F.softmax(self.forward2(out), dim=1)
        return out

    def figureOutFlattenShape(self, x):
        """Compute the flatten shape after conv layers"""
        out = F.relu(self.conv0(x))
        out = F.relu(self.conv1(out))
        out = self.mp0(out)
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.mp1(out)
        out = out.view(out.size(0), -1)
        return out.shape