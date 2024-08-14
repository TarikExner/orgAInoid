import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, img_size):
        super(SimpleCNN, self).__init__()
        
        self.img_x_dim = img_size
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout2d(0.3)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout2d(0.4)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout2d(0.5)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.feature_map_size = self.img_x_dim // 8
        
        self.fc1 = nn.Linear(64 * self.feature_map_size * self.feature_map_size, 128)
        self.dropout = nn.Dropout(0.75)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class SimpleCNNModel6(nn.Module):
    def __init__(self, img_x_dim):
        super(SimpleCNNModel6, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 64  # After six pooling layers, the size is divided by 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * self.feature_map_size * self.feature_map_size, 128)  # First hidden layer with 128 units
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer with 64 units
        self.fc3 = nn.Linear(64, 2)  # Output layer for binary classification
        
    def forward(self, x):
        # First convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Third convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Fourth convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv4(x)))
        
        # Fifth convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv5(x)))
        
        # Sixth convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv6(x)))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here if using CrossEntropyLoss
        
        return x

class SimpleCNNModel5(nn.Module):
    def __init__(self, img_x_dim):
        super(SimpleCNNModel5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 32  # After five pooling layers, the size is divided by 32
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * self.feature_map_size * self.feature_map_size, 128)  # First hidden layer with 128 units
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer with 64 units
        self.fc3 = nn.Linear(64, 2)  # Output layer for binary classification
        
    def forward(self, x):
        # First convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Third convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Fourth convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv4(x)))
        
        # Fifth convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv5(x)))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here if using CrossEntropyLoss
        
        return x

class SimpleCNNModel4(nn.Module):
    def __init__(self, img_x_dim):
        super(SimpleCNNModel4, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 16  # After four pooling layers, the size is divided by 16
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * self.feature_map_size * self.feature_map_size, 128)  # First hidden layer with 128 units
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer with 64 units
        self.fc3 = nn.Linear(64, 2)  # Output layer for binary classification
        
    def forward(self, x):
        # First convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Third convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Fourth convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here if using CrossEntropyLoss
        
        return x

class SimpleCNNModel3(nn.Module):
    def __init__(self, img_x_dim):
        super(SimpleCNNModel3, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 8  # After three pooling layers, the size is divided by 8
        
        # Fully connected layers (same as Model 1, 2, and 3)
        self.fc1 = nn.Linear(64 * self.feature_map_size * self.feature_map_size, 128)  # First hidden layer with 128 units
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer with 64 units
        self.fc3 = nn.Linear(64, 2)  # Output layer for binary classification
        
    def forward(self, x):
        # First convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Third convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU (same as Model 1, 2, and 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here if using CrossEntropyLoss
        
        return x

class SimpleCNNModel2(nn.Module):
    def __init__(self, img_x_dim):
        super(SimpleCNNModel2, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 4  # After two pooling layers, the size is divided by 4
        
        # Fully connected layers (same as Model 1 and Model 2)
        self.fc1 = nn.Linear(32 * self.feature_map_size * self.feature_map_size, 128)  # First hidden layer with 128 units
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer with 64 units
        self.fc3 = nn.Linear(64, 2)  # Output layer for binary classification
        
    def forward(self, x):
        # First convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU (same as Model 1 and Model 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here if using CrossEntropyLoss
        
        return x

class SimpleCNNModel1(nn.Module):
    def __init__(self, img_x_dim):
        super(SimpleCNNModel1, self).__init__()
        
        # Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layer
        self.feature_map_size = img_x_dim // 2  # After one pooling layer, the size is halved
        
        # Fully connected layers (same as Model 1)
        self.fc1 = nn.Linear(16 * self.feature_map_size * self.feature_map_size, 128)  # First hidden layer with 128 units
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer with 64 units
        self.fc3 = nn.Linear(64, 2)  # Output layer for binary classification
        
    def forward(self, x):
        # Convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU (same as Model 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here if using CrossEntropyLoss
        
        return x

class MLP(nn.Module):
    def __init__(self, img_x_dim):
        super(MLP, self).__init__()
        
        # Flatten the image into a vector
        self.input_size = img_x_dim * img_x_dim  # Flattened image size for grayscale image
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.input_size, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)               # Second hidden layer (optional)
        self.fc3 = nn.Linear(64, 2)                 # Output layer (2 classes for binary classification)
        
    def forward(self, x):
        # Flatten the image
        x = x.view(-1, self.input_size)
        
        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here if using CrossEntropyLoss
        
        return x