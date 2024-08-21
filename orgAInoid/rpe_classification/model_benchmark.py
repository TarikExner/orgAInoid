import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
from pathlib import Path
import os
import time

from .models import (MLP, SimpleCNNModel1, SimpleCNNModel2, SimpleCNNModel3, SimpleCNNModel4,
                     SimpleCNNModel5, SimpleCNNModel6, SimpleCNNModel7, SimpleCNNModel8, MLP_FC3,
                     SimpleCNNModel1_FC3, SimpleCNNModel2_FC3, SimpleCNNModel3_FC3, SimpleCNNModel4_FC3,
                     SimpleCNNModel5_FC3, SimpleCNNModel6_FC3, SimpleCNNModel7_FC3, SimpleCNNModel8_FC3)
from ._utils import ClassificationDataset
from ._dataset import read_classification_dataset

def hyperparameter_model_search_fc(experiment_id: str,
                                   output_dir = "./results_fc"):
    
    dataset = read_classification_dataset(f"./raw_data/{experiment_id}.cds")
    X_train, y_train, X_test, y_test = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
    
    models = [
        MLP_FC3, SimpleCNNModel1_FC3, SimpleCNNModel2_FC3, SimpleCNNModel3_FC3, SimpleCNNModel4_FC3,
        SimpleCNNModel5_FC3, SimpleCNNModel6_FC3, SimpleCNNModel7_FC3, SimpleCNNModel8_FC3
    ]
    
    # Learning rates to compare
    learning_rates = [0.01, 0.003, 0.001, 0.0003]

    batch_sizes = [32, 64, 128]

    class Unsqueezer:
        def __init__(self, transforms):
            self.transforms = transforms
        
        def __call__(self, img):
            img = img.squeeze(0)  # Remove the extra dimension to get [256, 256]
            img = self.transforms(img)  # Apply the transformations
            #img = img.unsqueeze(0)  # Add the channel dimension back to get [1, 256, 256]
            return img
    
    image_size = X_train.shape[2]
    transformations = Unsqueezer(transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(120),
        transforms.RandomResizedCrop(image_size, scale=(0.8,1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor()
    ]))
    
    # Number of epochs
    num_epochs = 100
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the CSV file with headers
    output_file = os.path.join(output_dir, f"model_search_{experiment_id}.csv")
    with open(output_file, "w") as file:
        file.write("ExperimentID,Model,LearningRate,BatchSize,Epoch,TrainLoss,ValLoss,TrainAccuracy,ValAccuracy,TrainF1,ValF1\n")
    
    # Main loop to iterate over models, learning rates, and transformations
    for model_class in models:
        model_start = time.time()
        for lr in learning_rates:
            for batch_size in batch_sizes:

                # Print current configuration
                print(f"[INFO] Starting experiment with Model: {model_class.__name__}, "
                      f"Learning Rate: {lr}, Batch Size: {batch_size}")
           
                # Initialize the dataset and dataloaders
                train_dataset = ClassificationDataset(X_train, y_train, transformations)
                val_dataset = ClassificationDataset(X_test, y_test, transformations)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                # Initialize the model, criterion, and optimizer
                model = model_class(image_size).to(device)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            
                # Training loop
                for epoch in range(num_epochs):
                    start = time.time()
                    model.train()
                    train_loss = 0
                    train_true = []
                    train_preds = []
                    
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output.squeeze(), target.float())
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        train_preds += torch.round(torch.sigmoid(output)).cpu().tolist()
                        train_true += target.cpu().tolist()
                    
                    train_loss /= len(train_loader)
                    train_acc = accuracy_score(train_true, train_preds)
                    train_f1 = f1_score(train_true, train_preds, average='macro')
                    
                    # Validation loop
                    model.eval()
                    val_loss = 0
                    val_true = []
                    val_preds = []
                    
                    with torch.no_grad():
                        for batch_idx, (data, target) in enumerate(val_loader):
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            loss = criterion(output.squeeze(), target.float())
                            
                            val_loss += loss.item()
                            val_preds += torch.round(torch.sigmoid(output)).cpu().tolist()
                            val_true += target.cpu().tolist()
                    
                    val_loss /= len(val_loader)
                    val_acc = accuracy_score(val_true, val_preds)
                    val_f1 = f1_score(val_true, val_preds, average='macro')

                    # Get the current learning rate before scheduler step
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # Step the learning rate scheduler based on the validation loss
                    scheduler.step(val_loss)
                    
                    # Check if the learning rate has been reduced
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr < current_lr:
                        print(f"[INFO] Learning rate reduced from {current_lr} to {new_lr}")
            
                    stop = time.time()
                    # Print metrics
                    print(f"[INFO] Epoch: {epoch+1}/{num_epochs}, "
                          f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                          f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, "
                          f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, "
                          f"Time: {stop-start}")
                    
                    # Write metrics to CSV file
                    with open(output_file, "a") as file:
                        file.write(f"{experiment_id},{model_class.__name__},{lr},"
                                   f"{batch_size},{epoch+1},{train_loss},{val_loss},"
                                   f"{train_acc},{val_acc},{train_f1},{val_f1}\n")
        model_stop = time.time()
        print("\nModel training took: ", model_stop - model_start, " seconds.\n")

def hyperparameter_model_search(experiment_id: str,
                                output_dir = "./results"):
    
    dataset = read_classification_dataset(f"./raw_data/{experiment_id}.cds")
    X_train, y_train, X_test, y_test = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
    
    models = [
        MLP, SimpleCNNModel1, SimpleCNNModel2, SimpleCNNModel3, SimpleCNNModel4,
        SimpleCNNModel5, SimpleCNNModel6, SimpleCNNModel7, SimpleCNNModel8
    ]
    
    # Learning rates to compare
    learning_rates = [0.01, 0.003, 0.001, 0.0003]

    batch_sizes = [32, 64, 128]

    class Unsqueezer:
        def __init__(self, transforms):
            self.transforms = transforms
        
        def __call__(self, img):
            img = img.squeeze(0)  # Remove the extra dimension to get [256, 256]
            img = self.transforms(img)  # Apply the transformations
            #img = img.unsqueeze(0)  # Add the channel dimension back to get [1, 256, 256]
            return img
    
    image_size = X_train.shape[2]
    transformations = Unsqueezer(transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(120),
        transforms.RandomResizedCrop(image_size, scale=(0.8,1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor()
    ]))
    
    # Number of epochs
    num_epochs = 100
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the CSV file with headers
    output_file = os.path.join(output_dir, f"model_search_{experiment_id}.csv")
    with open(output_file, "w") as file:
        file.write("ExperimentID,Model,LearningRate,BatchSize,Epoch,TrainLoss,ValLoss,TrainAccuracy,ValAccuracy,TrainF1,ValF1\n")
    
    # Main loop to iterate over models, learning rates, and transformations
    for model_class in models:
        model_start = time.time()
        for lr in learning_rates:
            for batch_size in batch_sizes:

                # Print current configuration
                print(f"[INFO] Starting experiment with Model: {model_class.__name__}, "
                      f"Learning Rate: {lr}, Batch Size: {batch_size}")
           
                # Initialize the dataset and dataloaders
                train_dataset = ClassificationDataset(X_train, y_train, transformations)
                val_dataset = ClassificationDataset(X_test, y_test, transformations)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                # Initialize the model, criterion, and optimizer
                model = model_class(image_size).to(device)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            
                # Training loop
                for epoch in range(num_epochs):
                    start = time.time()
                    model.train()
                    train_loss = 0
                    train_true = []
                    train_preds = []
                    
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output.squeeze(), target.float())
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        train_preds += torch.round(torch.sigmoid(output)).cpu().tolist()
                        train_true += target.cpu().tolist()
                    
                    train_loss /= len(train_loader)
                    train_acc = accuracy_score(train_true, train_preds)
                    train_f1 = f1_score(train_true, train_preds, average='macro')
                    
                    # Validation loop
                    model.eval()
                    val_loss = 0
                    val_true = []
                    val_preds = []
                    
                    with torch.no_grad():
                        for batch_idx, (data, target) in enumerate(val_loader):
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            loss = criterion(output.squeeze(), target.float())
                            
                            val_loss += loss.item()
                            val_preds += torch.round(torch.sigmoid(output)).cpu().tolist()
                            val_true += target.cpu().tolist()
                    
                    val_loss /= len(val_loader)
                    val_acc = accuracy_score(val_true, val_preds)
                    val_f1 = f1_score(val_true, val_preds, average='macro')

                    # Get the current learning rate before scheduler step
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # Step the learning rate scheduler based on the validation loss
                    scheduler.step(val_loss)
                    
                    # Check if the learning rate has been reduced
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr < current_lr:
                        print(f"[INFO] Learning rate reduced from {current_lr} to {new_lr}")
            
                    stop = time.time()
                    # Print metrics
                    print(f"[INFO] Epoch: {epoch+1}/{num_epochs}, "
                          f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                          f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, "
                          f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, "
                          f"Time: {stop-start}")
                    
                    # Write metrics to CSV file
                    with open(output_file, "a") as file:
                        file.write(f"{experiment_id},{model_class.__name__},{lr},"
                                   f"{batch_size},{epoch+1},{train_loss},{val_loss},"
                                   f"{train_acc},{val_acc},{train_f1},{val_f1}\n")
        model_stop = time.time()
        print("\nModel training took: ", model_stop - model_start, " seconds.\n")
                
