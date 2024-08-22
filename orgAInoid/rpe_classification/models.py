import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50, self).__init__()
        
        # Load the pre-trained ResNet50 model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Modify the final fully connected layer to match the number of output classes
        self.resnet50.fc = nn.Linear(in_features=2048, out_features=num_classes)
        
        # Determine if it's binary classification based on num_classes
        self.binary_classification = (num_classes == 2)
    
    def forward(self, x):
        x = self.resnet50(x)
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss


class VGG16_BN(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16_BN, self).__init__()
        
        # Load the pre-trained VGG16_BN model
        self.vgg16_bn = models.vgg16_bn(pretrained=True)
        
        # Modify the classifier to match the number of output classes
        self.vgg16_bn.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
        
        # Determine if it's binary classification based on num_classes
        self.binary_classification = (num_classes == 2)
    
    def forward(self, x):
        x = self.vgg16_bn(x)
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss



class DenseNet121(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNet121, self).__init__()
        
        # Load the pre-trained DenseNet121 model
        self.densenet121 = models.densenet121(pretrained=True)
        
        # Modify the final fully connected layer to match the number of output classes
        self.densenet121.classifier = nn.Linear(in_features=1024, out_features=num_classes)
        
        # Determine if it's binary classification based on num_classes
        self.binary_classification = (num_classes == 2)
    
    def forward(self, x):
        x = self.densenet121(x)
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss


class InceptionV3(nn.Module):
    def __init__(self, num_classes=2):
        super(InceptionV3, self).__init__()
        
        # Load the pre-trained InceptionV3 model
        self.inception_v3 = models.inception_v3(pretrained=True, aux_logits=False)
        
        # Modify the final fully connected layer to match the number of output classes
        self.inception_v3.fc = nn.Linear(in_features=2048, out_features=num_classes)
        
        # Determine if it's binary classification based on num_classes
        self.binary_classification = (num_classes == 2)
    
    def forward(self, x):
        x = self.inception_v3(x)
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV3_Large, self).__init__()
        
        # Load the pre-trained MobileNetV3-Large model
        self.mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
        
        # Modify the final classifier to match the number of output classes
        self.mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes)
        
        # Determine if it's binary classification based on num_classes
        self.binary_classification = (num_classes == 2)
    
    def forward(self, x):
        x = self.mobilenet_v3_large(x)
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class SimpleCNNModel8_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel8_FC3, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 256  # After eight pooling layers, the size is divided by 256
        
        # Fully connected layers with symmetrical structure
        self.fc1 = nn.Linear(2048 * self.feature_map_size * self.feature_map_size, 256)  # First hidden layer with 256 units
        self.fc2 = nn.Linear(256, 128)  # Second hidden layer with 128 units
        self.fc3 = nn.Linear(128, 64)  # Third hidden layer with 64 units
        self.fc4 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
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
        
        # Seventh convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv7(x)))
        
        # Eighth convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv8(x)))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here if using CrossEntropyLoss
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class SimpleCNNModel8(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel8, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 256  # After eight pooling layers, the size is divided by 256
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048 * self.feature_map_size * self.feature_map_size, 128)  # First hidden layer with 128 units
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer with 64 units
        self.fc3 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
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
        
        # Seventh convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv7(x)))
        
        # Eighth convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv8(x)))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here if using CrossEntropyLoss
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss


class SimpleCNNModel7_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel7_FC3, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 128  # After seven pooling layers, the size is divided by 128
        
        # Fully connected layers with symmetrical structure
        self.fc1 = nn.Linear(1024 * self.feature_map_size * self.feature_map_size, 256)  # First hidden layer with 256 units
        self.fc2 = nn.Linear(256, 128)  # Second hidden layer with 128 units
        self.fc3 = nn.Linear(128, 64)  # Third hidden layer with 64 units
        self.fc4 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
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
        
        # Seventh convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv7(x)))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here if using CrossEntropyLoss
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss


class SimpleCNNModel7(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel7, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 128  # After seven pooling layers, the size is divided by 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024 * self.feature_map_size * self.feature_map_size, 128)  # First hidden layer with 128 units
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer with 64 units
        self.fc3 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
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
        
        # Seventh convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv7(x)))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here if using CrossEntropyLoss
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class SimpleCNNModel6_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel6_FC3, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 64  # After six pooling layers, the size is divided by 64
        
        # Fully connected layers with symmetrical structure
        self.fc1 = nn.Linear(512 * self.feature_map_size * self.feature_map_size, 256)  # First hidden layer with 256 units
        self.fc2 = nn.Linear(256, 128)  # Second hidden layer with 128 units
        self.fc3 = nn.Linear(128, 64)  # Third hidden layer with 64 units
        self.fc4 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
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
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here if using CrossEntropyLoss
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class SimpleCNNModel6(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel6, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
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
        self.fc3 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
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
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class SimpleCNNModel5_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel5_FC3, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 32  # After five pooling layers, the size is divided by 32
        
        # Fully connected layers with symmetrical structure
        self.fc1 = nn.Linear(256 * self.feature_map_size * self.feature_map_size, 256)  # First hidden layer with 256 units
        self.fc2 = nn.Linear(256, 128)  # Second hidden layer with 128 units
        self.fc3 = nn.Linear(128, 64)  # Third hidden layer with 64 units
        self.fc4 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
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
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here if using CrossEntropyLoss
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class SimpleCNNModel5(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel5, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
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
        self.fc3 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
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
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class SimpleCNNModel4_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel4_FC3, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 16  # After four pooling layers, the size is divided by 16
        
        # Fully connected layers with symmetrical structure
        self.fc1 = nn.Linear(128 * self.feature_map_size * self.feature_map_size, 256)  # First hidden layer with 256 units
        self.fc2 = nn.Linear(256, 128)  # Second hidden layer with 128 units
        self.fc3 = nn.Linear(128, 64)  # Third hidden layer with 64 units
        self.fc4 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
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
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here if using CrossEntropyLoss
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class SimpleCNNModel4(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel4, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
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
        self.fc3 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
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
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class SimpleCNNModel3_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel3_FC3, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 8  # After three pooling layers, the size is divided by 8
        
        # Fully connected layers with symmetrical structure
        self.fc1 = nn.Linear(64 * self.feature_map_size * self.feature_map_size, 256)  # First hidden layer with 256 units
        self.fc2 = nn.Linear(256, 128)  # Second hidden layer with 128 units
        self.fc3 = nn.Linear(128, 64)  # Third hidden layer with 64 units
        self.fc4 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
    def forward(self, x):
        # First convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Third convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here if using CrossEntropyLoss
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class SimpleCNNModel3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel3, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 8  # After three pooling layers, the size is divided by 8
        
        # Fully connected layers (same as Model 1, 2, and 3)
        self.fc1 = nn.Linear(64 * self.feature_map_size * self.feature_map_size, 128)  # First hidden layer with 128 units
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer with 64 units
        self.fc3 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
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
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class SimpleCNNModel2_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel2_FC3, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 4  # After two pooling layers, the size is divided by 4
        
        # Fully connected layers with symmetrical structure
        self.fc1 = nn.Linear(32 * self.feature_map_size * self.feature_map_size, 256)  # First hidden layer with 256 units
        self.fc2 = nn.Linear(256, 128)  # Second hidden layer with 128 units
        self.fc3 = nn.Linear(128, 64)  # Third hidden layer with 64 units
        self.fc4 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
    def forward(self, x):
        # First convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here if using CrossEntropyLoss
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class SimpleCNNModel2(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel2, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layers
        self.feature_map_size = img_x_dim // 4  # After two pooling layers, the size is divided by 4
        
        # Fully connected layers (same as Model 1 and Model 2)
        self.fc1 = nn.Linear(32 * self.feature_map_size * self.feature_map_size, 128)  # First hidden layer with 128 units
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer with 64 units
        self.fc3 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
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
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class SimpleCNNModel1_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel1_FC3, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layer
        self.feature_map_size = img_x_dim // 2  # After one pooling layer, the size is halved
        
        # Fully connected layers with symmetrical structure
        self.fc1 = nn.Linear(16 * self.feature_map_size * self.feature_map_size, 256)  # First hidden layer with 256 units
        self.fc2 = nn.Linear(256, 128)  # Second hidden layer with 128 units
        self.fc3 = nn.Linear(128, 64)  # Third hidden layer with 64 units
        self.fc4 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
    def forward(self, x):
        # Convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here if using CrossEntropyLoss
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class SimpleCNNModel1(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel1, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after the pooling layer
        self.feature_map_size = img_x_dim // 2  # After one pooling layer, the size is halved
        
        # Fully connected layers (same as Model 1)
        self.fc1 = nn.Linear(16 * self.feature_map_size * self.feature_map_size, 128)  # First hidden layer with 128 units
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer with 64 units
        self.fc3 = nn.Linear(64, num_classes)  # Output layer for binary classification
        
    def forward(self, x):
        # Convolutional layer with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU (same as Model 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here if using CrossEntropyLoss
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss


class MLP_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(MLP_FC3, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Flatten the image into a vector
        self.input_size = img_x_dim * img_x_dim  # Flattened image size for grayscale image
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.input_size, 256)  # First hidden layer with 256 units
        self.fc2 = nn.Linear(256, 128)  # Second hidden layer with 128 units
        self.fc3 = nn.Linear(128, 64)  # Third hidden layer with 64 units
        self.fc4 = nn.Linear(64, num_classes)  # Output layer (2 classes for binary classification)
        
    def forward(self, x):
        # Flatten the image
        x = x.view(-1, self.input_size)
        
        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here if using CrossEntropyLoss
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss

class MLP(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(MLP, self).__init__()

        self.binary_classification = (num_classes == 2)
        
        # Flatten the image into a vector
        self.input_size = img_x_dim * img_x_dim  # Flattened image size for grayscale image
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.input_size, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)               # Second hidden layer (optional)
        self.fc3 = nn.Linear(64, num_classes)                 # Output layer (2 classes for binary classification)
        
    def forward(self, x):
        # Flatten the image
        x = x.view(-1, self.input_size)
        
        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here if using CrossEntropyLoss
        
        # Apply sigmoid for binary classification, softmax for multi-class classification
        if self.binary_classification:
            return x  # Return logits for BCEWithLogitsLoss
        else:
            return torch.softmax(x, dim=1)  # Apply softmax for CrossEntropyLoss
