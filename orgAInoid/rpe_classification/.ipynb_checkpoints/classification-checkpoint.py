import pandas as pd
import numpy as np
from pathlib import Path
import torch
import os
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import time

from tqdm import tqdm

import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score, f1_score

from ..segmentation.model import UNet
from .models import SimpleCNN
from ._utils import _prepare_classification_dataset, ClassificationDataset

from .._utils import _read_image, _get_masked_image, _load_unet_model

def run_classification(classification_id: str,
                       file_frame: pd.DataFrame,
                       start_timepoint: int,
                       stop_timepoint: int,
                       image_size: int = 256,
                       unet_dir: Path = "../segmentation",
                       unet_input_size: int = 256,
                       experiment_dir: Path = "../../"):

    """
    The file frame contains all images with annotations. Its supplied
    in order to define the complete training dataset and can contain
    multiple experiments or whatever is supposed to be done.

    First, the wells corresponding to the RPE annotations and experiment_ids
    are selected. Based on them, a train_test_split is defined and the
    images are assembled.

    experiment_dir refers to the directory where the experiment folders are stored.
    This is important in order to find the correct image Paths.

    The classification ID is meant to mark different experiments with their unique
    .config file (binary encoded!). Since we dont have to reassemble the dataset again, we'll be able
    to test a lot more conditions.

    """
    # timepoints
    BATCH_SIZE = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if DEVICE == "cuda" else False
    INIT_LR = 0.001
    NUM_EPOCHS = 100
    
    X_train, y_train, X_test, y_test = _prepare_classification_dataset(classification_id = classification_id,
                                                                       file_frame = file_frame,
                                                                       start_timepoint = start_timepoint,
                                                                       stop_timepoint = stop_timepoint,
                                                                       image_size = image_size,
                                                                       unet_dir = unet_dir,
                                                                       unet_input_size = unet_input_size,
                                                                       experiment_dir = experiment_dir)

    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),    # Randomly flip the image vertically
        transforms.RandomRotation(120),      # Randomly rotate the image by up to 15 degrees
        transforms.RandomResizedCrop(image_size, scale=(0.8,1.2)),  # Randomly crop and resize to image size
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, etc.
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Apply Gaussian Blur
        transforms.ToTensor()               # Convert PIL image to Tensor
    ])
    
    transformations_test = transforms.Compose([
        transforms.ToPILImage(),
     	transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    train_dataset = ClassificationDataset(
        image_arr = X_train,
        classes = y_train,
    	transforms = transformations
    )
    test_dataset = ClassificationDataset(
        image_arr = X_test,
        classes = y_test,
        transforms = transformations_test
    )
    
    train_loader = DataLoader(
        train_dataset,
        shuffle = True,
    	batch_size = BATCH_SIZE,
        pin_memory = PIN_MEMORY,
    	num_workers = 1
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle = False,
    	batch_size = BATCH_SIZE,
        pin_memory = PIN_MEMORY,
    	num_workers = 1
    )

    cnn_model = SimpleCNN(img_size = image_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cnn_model.to(device)
    
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=INIT_LR, weight_decay=1e-4)

    scheduler_plateau = ReduceLROnPlateau(optimizer, 'min', patience = 10, verbose = True, threshold = 0.0001, factor = 0.5)
    scheduler_step = StepLR(optimizer, step_size=30, gamma=0.5)
    
    trainSteps = len(train_dataset) // BATCH_SIZE
    testSteps = len(test_dataset) // BATCH_SIZE
    
    H = {"train_loss": [], "test_loss": [], "accuracy": [], "f1": [], "epoch": []}
    best_val_loss = float('inf')

    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(NUM_EPOCHS)):
        model.train()
        train_loss = 0.0
        
        for (i, (images, classes)) in enumerate(train_loader):
            (images, classes) = (images.to(DEVICE), classes.to(DEVICE))
            predictions = model(images)
            loss = criterion(predictions, classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss

        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            model.eval()
            for (images, classes) in test_loader:
                (images, classes) = (images.to(DEVICE), classes.to(DEVICE))
                comp_classes = torch.argmax(classes, dim=1)
                val_outputs = model(images)
                val_batch_loss = criterion(val_outputs, classes)
                val_loss += val_batch_loss
                
                # Collect validation predictions and labels
                _, val_batch_preds = torch.max(val_outputs, 1)
                val_preds.extend(val_batch_preds.cpu().numpy())
                val_labels.extend(comp_classes.cpu().numpy())
        # Calculate validation metrics

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        print("F1: ", val_f1)
        print("Accuracy: ", val_acc)
                
        avg_train_loss = train_loss / trainSteps
        avg_val_loss = val_loss / testSteps
        current_lr = optimizer.param_groups[0]['lr']
        
        # Step the ReduceLROnPlateau scheduler
        scheduler_plateau.step(avg_val_loss)
        
        # Step the StepLR scheduler
        #scheduler_step.step()
        
        # Check and print if the learning rate has changed
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"Learning rate decreased from {current_lr:.6f} to {new_lr:.6f}")
    
        H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        H["test_loss"].append(avg_val_loss.cpu().detach().numpy())
        H["accuracy"].append(val_acc)
        H["f1"].append(val_f1)
        H["epoch"].append(e+1)
        print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avg_train_loss, avg_val_loss))
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = e+1
            print(f"Saving model with validation loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), f"{classification_id}_best_model.pth")

    results = pd.DataFrame(H)
    results["batch_size"] = BATCH_SIZE

    results.to_csv(f"{classification_id}_losses.txt", index = False)
    
    endTime = time.time()
    print(f"[INFO] total time taken to train the model: {endTime - startTime}")
    print(f"[INFO] the best val_loss was: {best_val_loss} and was calculated in epoch {best_epoch}")


    
    return H


if __name__ == "__main__":
    run_classification()