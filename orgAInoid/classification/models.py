import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import (ResNet50_Weights,
                                VGG16_BN_Weights,
                                DenseNet121_Weights,
                                MobileNet_V3_Large_Weights)


class ResNet50(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5, **kwargs):
        super(ResNet50, self).__init__()
        
        # Load the pre-trained ResNet50 model
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT, **kwargs)
        
        # Modify the final fully connected layer to match the number of output classes
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(p=dropout),  # Add dropout layer with p=0.5
            nn.Linear(in_features=2048, out_features=num_classes)  # Final linear layer
        )       
        
    def forward(self, x):
        x = self.resnet50(x)
        return x

    def freeze_layers(self, freeze_layers):
        """
        Freeze layers of the model up to a certain depth.
        
        Parameters:
        - freeze_layers (int): Number of layers to freeze. 
                               E.g., 0 = freeze all except 'fc', 1 = freeze up to 'conv1', etc.
                               If -1, freeze all layers except the final FC layer.
                               If -2, freeze all layers except the last two layers.
                               If -N, freeze all layers except the last N layers.
        """
        layer_mapping = {
            0: ['conv1'],
            1: ['bn1'],
            2: ['relu'],
            3: ['maxpool'],
            4: ['layer1'],
            5: ['layer2'],
            6: ['layer3'],
            7: ['layer4'],
            8: ['avgpool'],
            9: ['fc']  # This is the output layer, so it should not be frozen.
        }

        # Total number of layers in the model
        total_layers = len(layer_mapping)

        # Check if freeze_layers is within a valid range
        if abs(freeze_layers) > total_layers:
            raise ValueError(f"freeze_layers cannot be less than {-total_layers} or greater than {total_layers - 1}")

        # If freeze_layers is negative, calculate how many layers to unfreeze
        if freeze_layers < 0:
            unfreeze_from = total_layers + freeze_layers  # e.g., freeze_layers = -1 will unfreeze from layer 8
        else:
            unfreeze_from = freeze_layers + 1  # If freeze_layers is positive, we freeze up to that layer

        # Freeze layers up to the specified index or negative offset
        for i, (_, layer) in enumerate(self.resnet50.named_children()):
            if i < unfreeze_from:
                for param in layer.parameters():
                    param.requires_grad = False

        # Ensure the final fully connected layer is always trainable
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True

class VGG16_BN(nn.Module):
    def __init__(self, num_classes=2, dropout = 0.5, **kwargs):
        super(VGG16_BN, self).__init__()
        
        # Load the pre-trained VGG16_BN model
        self.vgg16_bn = models.vgg16_bn(
            weights=VGG16_BN_Weights.DEFAULT,
            dropout = dropout,
            **kwargs
        )
        
        # Modify the classifier to match the number of output classes
        self.vgg16_bn.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
        
    def forward(self, x):
        x = self.vgg16_bn(x)
        return x
        
    def freeze_layers(self, freeze_layers):
        """
        Freeze layers of the model up to a certain depth.
        
        Parameters:
        - freeze_layers (int): Number of layers to freeze, referring to the printed model representation.
                               If -1, freeze all layers except the final FC layer.
                               If -2, freeze all layers except the last two layers.
                               If -N, freeze all layers except the last N layers.
                               If the value exceeds the number of layers, raise ValueError.
        """
        # Layer mapping for VGG16_BN features
        layer_mapping = {
            0: ['features.0', 'features.1', 'features.2'],
            1: ['features.3', 'features.4', 'features.5'],
            2: ['features.6'],
            3: ['features.7', 'features.8', 'features.9'],
            4: ['features.10', 'features.11', 'features.12'],
            5: ['features.13'],
            6: ['features.14', 'features.15', 'features.16'],
            7: ['features.17', 'features.18', 'features.19'],
            8: ['features.20', 'features.21', 'features.22'],
            9: ['features.23'],
            10: ['features.24', 'features.25', 'features.26'],
            11: ['features.27', 'features.28', 'features.29'],
            12: ['features.30', 'features.31', 'features.32'],
            13: ['features.33'],
            14: ['features.34', 'features.35', 'features.36'],
            15: ['features.37', 'features.38', 'features.39'],
            16: ['features.40', 'features.41', 'features.42'],
            17: ['features.43']  # Last layer of features
        }

        # Total number of layers in the model
        total_layers = len(layer_mapping) + 1  # Add 1 for classifier

        # Check if freeze_layers is within a valid range
        if abs(freeze_layers) > total_layers:
            raise ValueError(f"freeze_layers cannot be less than {-total_layers} or greater than {total_layers - 1}")

        # If freeze_layers is negative, calculate how many layers to unfreeze
        if freeze_layers < 0:
            unfreeze_from = total_layers + freeze_layers  # e.g., freeze_layers = -1 will unfreeze from layer 17 (classifier)
        else:
            unfreeze_from = freeze_layers + 1  # If freeze_layers is positive, we freeze up to that layer

        # Freeze layers up to the specified index or negative offset
        for i, (_, layer) in enumerate(self.vgg16_bn.named_children()):
            if i < unfreeze_from:
                for param in layer.parameters():
                    param.requires_grad = False

        # Ensure the final classifier layer is always trainable
        for param in self.vgg16_bn.classifier.parameters():
            param.requires_grad = True

class DenseNet121(nn.Module):
    def __init__(self, num_classes=2, dropout = 0.2, **kwargs):
        super(DenseNet121, self).__init__()
        
        # Load the pre-trained DenseNet121 model
        self.densenet121 = models.densenet121(weights=DenseNet121_Weights.DEFAULT, **kwargs)
        
        # Modify the final fully connected layer to match the number of output classes
        self.densenet121.classifier = nn.Sequential(
            nn.Dropout(p=dropout),  # Add dropout layer with p=0.5
            nn.Linear(in_features=1024, out_features=num_classes)  # Final linear layer
        )       
    
    def forward(self, x):
        x = self.densenet121(x)
        return x
        
    def freeze_layers(self, freeze_layers):
        """
        Freeze layers of the model up to a certain depth.
        
        Parameters:
        - freeze_layers (int): Number of layers to freeze, referring to the printed model representation.
                               If -1, freeze all layers except the final FC layer.
                               If -2, freeze all layers except the last two layers.
                               If -N, freeze all layers except the last N layers.
                               If the value exceeds the number of layers, raise ValueError.
        """
        # Define the layer mapping based on the architecture
        layer_mapping = {
            0: ['conv0', 'norm0', 'relu0', 'pool0'],
            1: ['denseblock1'],
            2: ['transition1'],
            3: ['denseblock2'],
            4: ['transition2'],
            5: ['denseblock3'],
            6: ['transition3'],
            7: ['denseblock4'],
            8: ['norm5']  # Last layer before the classifier
        }

        # Total number of layers in DenseNet121
        total_layers = len(layer_mapping) + 1  # Add 1 for the classifier

        # Check if freeze_layers is within a valid range
        if abs(freeze_layers) > total_layers:
            raise ValueError(f"freeze_layers cannot be less than {-total_layers} or greater than {total_layers - 1}")

        # If freeze_layers is negative, calculate how many layers to unfreeze
        if freeze_layers < 0:
            unfreeze_from = total_layers + freeze_layers  # e.g., freeze_layers = -1 will unfreeze from norm5 (classifier remains unfrozen)
        else:
            unfreeze_from = freeze_layers + 1  # If freeze_layers is positive, freeze up to and including that index

        # Convert the model's layers to a list (focus only on feature layers)
        layers = list(self.densenet121.features.named_children())

        # Freeze layers up to the specified index
        for i, (_, layer) in enumerate(layers):
            if i < unfreeze_from:
                for param in layer.parameters():
                    param.requires_grad = False

        # Ensure the final classifier layer remains trainable
        for param in self.densenet121.classifier.parameters():
            param.requires_grad = True

class InceptionV3(nn.Module):
    def __init__(self, num_classes=2):
        super(InceptionV3, self).__init__()
        
        # Load the pre-trained InceptionV3 model
        self.inception_v3 = models.inception_v3(pretrained=True, aux_logits=True)
        
        # Modify the final fully connected layer to match the number of output classes
        self.inception_v3.fc = nn.Linear(in_features=2048, out_features=num_classes)
        
    def forward(self, x):
        x = self.inception_v3(x)
        return x
        

class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=2, dropout = 0.5, **kwargs):
        super(MobileNetV3_Large, self).__init__()

        # Load the pre-trained MobileNetV3-Large model
        self.mobilenet_v3_large = models.mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.DEFAULT,
            dropout = dropout,
            **kwargs
        )
        
        # Modify the final classifier to match the number of output classes
        self.mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes)
        
    def forward(self, x):
        x = self.mobilenet_v3_large(x)
        return x

    def freeze_layers(self, freeze_layers):
        """
        Freeze layers of the model up to a certain depth.

        Parameters:
        - freeze_layers (int): Number of layers to freeze, referring to the printed model representation.
                               If -1, freeze all layers except the final FC layer.
                               If -2, freeze all layers except the last two layers.
                               If the value exceeds the number of layers, raise ValueError.
        """
        # Define the layer mapping based on the architecture
        layer_mapping = {
            0: ['features', '0'],  # Conv2dNormActivation (initial convolution)
            1: ['features', '1'],  # InvertedResidual block 1
            2: ['features', '2'],  # InvertedResidual block 2
            3: ['features', '3'],  # InvertedResidual block 3
            4: ['features', '4'],  # InvertedResidual block 4
            5: ['features', '5'],  # InvertedResidual block 5
            6: ['features', '6'],  # InvertedResidual block 6
            7: ['features', '7'],  # InvertedResidual block 7
            8: ['features', '8'],  # InvertedResidual block 8
            9: ['features', '9'],  # InvertedResidual block 9
            10: ['features', '10'],  # InvertedResidual block 10
            11: ['features', '11'],  # InvertedResidual block 11
            12: ['features', '12'],  # InvertedResidual block 12
            13: ['features', '13'],  # InvertedResidual block 13
            14: ['features', '14'],  # InvertedResidual block 14
            15: ['features', '15'],  # InvertedResidual block 15
            16: ['features', '16'],  # Conv2dNormActivation (final convolution before pooling)
        }

        # Total number of layers in MobileNetV3_Large
        total_layers = len(layer_mapping) + 1  # +1 for the classifier

        # Check if freeze_layers is within a valid range
        if abs(freeze_layers) > total_layers:
            raise ValueError(f"freeze_layers cannot be less than {-total_layers} or greater than {total_layers - 1}")

        # If freeze_layers is negative, calculate how many layers to unfreeze
        if freeze_layers < 0:
            unfreeze_from = total_layers + freeze_layers  # e.g., freeze_layers = -1 will unfreeze the classifier layer
        else:
            unfreeze_from = freeze_layers + 1  # If freeze_layers is positive, freeze up to and including that index

        # Convert the model's layers to a list (focus on feature layers)
        layers = list(self.mobilenet_v3_large.features.named_children())

        # Freeze layers up to the specified index
        for i, (name, layer) in enumerate(layers):
            if i < unfreeze_from:
                for param in layer.parameters():
                    param.requires_grad = False

        # Ensure the final classifier layer remains trainable
        for param in self.mobilenet_v3_large.classifier.parameters():
            param.requires_grad = True


class SimpleCNNModel8_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel8_FC3, self).__init__()
        
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
        self.feature_map_size = img_x_dim // 128  # After seven pooling layers, the size is divided by 128
        
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
        
        # Eighth convolutional layer without pooling
        x = F.relu(self.conv8(x))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here if using CrossEntropyLoss

        return x
        

class SimpleCNNModel8(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel8, self).__init__()

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
        self.feature_map_size = img_x_dim // 128 # After seven pooling layers, the size is divided by 128
        
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
        
        # Eighth convolutional layer without pooling
        x = F.relu(self.conv8(x))
        
        # Flatten the tensor while preserving the batch size
        x = x.view(x.size(0), -1)  # Flattening
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here if using CrossEntropyLoss

        return x

class SimpleCNNModel7_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel7_FC3, self).__init__()

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
        
        return x

class SimpleCNNModel7(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel7, self).__init__()

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

        return x

class SimpleCNNModel6_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel6_FC3, self).__init__()

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

        return x

class SimpleCNNModel6(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel6, self).__init__()

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

        return x

class SimpleCNNModel5_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel5_FC3, self).__init__()

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

        return x

class SimpleCNNModel5(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel5, self).__init__()

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

        return x


class SimpleCNNModel4_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel4_FC3, self).__init__()

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

        return x


class SimpleCNNModel4(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel4, self).__init__()

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

        return x


class SimpleCNNModel3_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel3_FC3, self).__init__()

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

        return x


class SimpleCNNModel3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel3, self).__init__()

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

        return x


class SimpleCNNModel2_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel2_FC3, self).__init__()

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

        return x


class SimpleCNNModel2(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel2, self).__init__()

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

        return x


class SimpleCNNModel1_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel1_FC3, self).__init__()

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
        
        return x


class SimpleCNNModel1(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(SimpleCNNModel1, self).__init__()

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
        
        return x


class MLP_FC3(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(MLP_FC3, self).__init__()

        # Flatten the image into a vector
        self.input_size = img_x_dim * img_x_dim * 3  # Flattened image size for grayscale image
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.input_size, 256)  # First hidden layer with 256 units
        self.fc2 = nn.Linear(256, 128)  # Second hidden layer with 128 units
        self.fc3 = nn.Linear(128, 64)  # Third hidden layer with 64 units
        self.fc4 = nn.Linear(64, num_classes)  # Output layer (2 classes for binary classification)
        
    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here if using CrossEntropyLoss

        return x


class MLP(nn.Module):
    def __init__(self, img_x_dim = 224, num_classes = 2):
        super(MLP, self).__init__()

        # Flatten the image into a vector
        self.input_size = img_x_dim * img_x_dim * 3  # Flattened image size for grayscale image
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.input_size, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)               # Second hidden layer (optional)
        self.fc3 = nn.Linear(64, num_classes)                 # Output layer (2 classes for binary classification)
        
    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here if using CrossEntropyLoss

        return x
