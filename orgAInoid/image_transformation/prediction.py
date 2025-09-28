import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from sklearn.model_selection import train_test_split
from .models import ImageTransformer, ImageTransformerV2
from matplotlib import pyplot as plt
from typing import Literal
from pathlib import Path
import time

import os


def plot_current_state(
    images: np.ndarray,
    target: np.ndarray,
    model,
    epoch: int,
    device: str,
    experiment_id: str,
    image_output_dir: Path,
    image_index: int = 0,
):
    img_series = images[image_index, :, :, :]
    img_series = img_series.detach().cpu().numpy()

    prediction = model(images.to(device))
    prediction = prediction[image_index, :, :, :]
    prediction = prediction.detach().cpu().numpy()

    target_image = target[image_index, :, :, :]
    target_image = target_image.detach().cpu().numpy()

    raw_imgs = np.vstack([img_series, target_image])
    pred_imgs = np.vstack([img_series, prediction])

    fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(12, 4))
    ax = ax.reshape(2, 6)
    for i, img in enumerate(raw_imgs):
        ax[0, i].imshow(img, cmap="Greys_r")
    for i, img in enumerate(pred_imgs):
        ax[1, i].imshow(img, cmap="Greys_r")
    ax = ax.flatten()
    for axis in ax:
        axis.tick_params(left=False, bottom=False)
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    ax = ax.reshape(2, 6)
    ax[0, 5].set_title("Target")
    ax[1, 5].set_title("Predicted")
    plt.suptitle(f"Epoch: {epoch}")
    plt.savefig(
        os.path.join(image_output_dir, f"{experiment_id}_epoch_{epoch}.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


class ImageSequenceDataset(Dataset):
    def __init__(self, image_sequences, subseries_length, prediction_length=1):
        """
        Args:
            image_sequences (torch.Tensor): The full dataset of shape [num_sequences, num_timepoints, height, width].
            subseries_length (int): The length of each sub-series to be used for prediction.
            prediction_length (int): The number of future timepoints to predict.
        """
        self.image_sequences = image_sequences
        self.subseries_length = subseries_length
        self.prediction_length = prediction_length
        self.num_sequences = image_sequences.shape[0]
        self.num_timepoints = image_sequences.shape[1]

    def __len__(self):
        # Total number of sub-series in all sequences, accounting for the prediction length
        return (
            self.num_timepoints - self.subseries_length - self.prediction_length + 1
        ) * self.num_sequences

    def __getitem__(self, idx):
        sequence_idx = idx // (
            self.num_timepoints - self.subseries_length - self.prediction_length + 1
        )
        timepoint_idx = idx % (
            self.num_timepoints - self.subseries_length - self.prediction_length + 1
        )

        subseries = self.image_sequences[
            sequence_idx, timepoint_idx : timepoint_idx + self.subseries_length
        ]
        target = self.image_sequences[
            sequence_idx,
            timepoint_idx + self.subseries_length : timepoint_idx
            + self.subseries_length
            + self.prediction_length,
        ]

        # Ensure the target has the correct shape: [prediction_length, 1, height, width]
        # target = target.unsqueeze(1)  # Add channel dimension: [prediction_length, 1, height, width]

        return subseries, target


def run_prediction_v2(
    experiment_id: str,
    subseries_length: int = 5,
    prediction_length: int = 10,
    batch_size: int = 64,
    img_size: tuple[int] = (128, 128),
    patch_size: int = 16,
    num_layers: int = 6,
    num_heads: int = 8,
    embed_dim: int = 512,
    n_epochs: int = 200,
    forward_expansion: int = 2048,
    dropout: float = 0.1,
    mode: Literal["training", "gif_generation"] = "training",
    model_output_dir: Path = "./models",
    image_output_dir: Path = "./images",
    from_state_dict: str = "",
):
    e001 = np.load("../raw_data/timeseries_E001_128.npy")
    e004 = np.load("../raw_data/timeseries_E004_128.npy")
    e005 = np.load("../raw_data/timeseries_E005_128.npy")
    e006 = np.load("../raw_data/timeseries_E006_128.npy")

    data = np.vstack([e001, e004, e005, e006])

    X_train, X_test = train_test_split(data, test_size=0.1, random_state=187)

    subseries_length = subseries_length

    train_data = ImageSequenceDataset(
        X_train, subseries_length, prediction_length=prediction_length
    )
    test_data = ImageSequenceDataset(
        X_test, subseries_length, prediction_length=prediction_length
    )

    batch_size = batch_size

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    img_size = img_size
    patch_size = patch_size
    num_layers = num_layers
    num_heads = num_heads
    embed_dim = embed_dim
    forward_expansion = forward_expansion
    dropout = dropout

    model = ImageTransformerV2(
        img_size,
        patch_size,
        subseries_length,
        embed_dim,
        num_heads,
        num_layers,
        dropout,
        prediction_length=prediction_length,
    )
    if from_state_dict:
        state_dict = torch.load(from_state_dict)
        model.load_state_dict(state_dict, strict=False)
        print("initialized previous weights")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if mode == "gif_generation":
        torch.save(
            model.state_dict(),
            os.path.join(model_output_dir, f"{experiment_id}_epoch_0.pth"),
        )

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    score_file = f"losses_{experiment_id}.txt"
    if not os.path.isfile(score_file):
        print("initializing score file")
        with open(score_file, "w") as f:
            f.write(
                "epoch,train_loss,val_loss,batch_size,img_size,patch_size,num_layers,num_heads,embed_dim,forward_expansion,dropout,subseries_length,prediction_length\n"
            )

    # Define the learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Number of epochs
    num_epochs = n_epochs
    best_val_loss = float("inf")
    # Training loop
    for epoch in range(num_epochs):
        start = time.time()
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            # Move data to the appropriate device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Compute average training loss
        avg_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():  # No need to track gradients during validation
            for i, (inputs, targets) in enumerate(val_loader):
                # Move data to the appropriate device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # if epoch%10 == 0 and i==0:
                #    plot_current_state(inputs, targets, model, epoch, device, experiment_id, image_output_dir)

                # Compute loss
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            print(f"Saving model with validation loss: {best_val_loss:.4f}")
            torch.save(
                model.state_dict(),
                os.path.join(model_output_dir, f"{experiment_id}_best_model.pth"),
            )

        if mode == "gif_generation":
            torch.save(
                model.state_dict(),
                os.path.join(
                    model_output_dir, f"{experiment_id}_epoch_{epoch + 1}.pth"
                ),
            )

        with open(score_file, "a") as f:
            (
                img_size,
                patch_size,
                num_layers,
                num_heads,
                embed_dim,
                forward_expansion,
                dropout,
            )
            f.write(
                f"{epoch + 1},{avg_train_loss},{avg_val_loss},{batch_size},{img_size[0]},{patch_size},{num_layers},{num_heads},{embed_dim},{forward_expansion},{dropout},{subseries_length},{prediction_length}\n"
            )

        current_lr = optimizer.param_groups[0]["lr"]
        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Check and print if the learning rate has changed
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            print(f"Learning rate decreased from {current_lr:.6f} to {new_lr:.6f}")
        # Print training and validation loss
        stop = time.time()
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Training Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f},"
            f"Time Spent: {stop - start:.2f}"
        )
    print("Training complete!")


def run_prediction(
    experiment_id: str,
    subseries_length: int = 5,
    batch_size: int = 64,
    img_size: tuple[int] = (128, 128),
    patch_size: int = 16,
    num_layers: int = 6,
    num_heads: int = 8,
    embed_dim: int = 512,
    n_epochs: int = 200,
    forward_expansion: int = 2048,
    dropout: float = 0.1,
    mode: Literal["training", "gif_generation"] = "training",
    model_output_dir: Path = "./models",
    image_output_dir: Path = "./images",
):
    e001 = np.load("../raw_data/timeseries_E001_128.npy")
    e004 = np.load("../raw_data/timeseries_E004_128.npy")
    e005 = np.load("../raw_data/timeseries_E005_128.npy")
    e006 = np.load("../raw_data/timeseries_E006_128.npy")

    data = np.vstack([e001, e004, e005, e006])

    X_train, X_test = train_test_split(data, test_size=0.1, random_state=187)

    subseries_length = subseries_length

    train_data = ImageSequenceDataset(X_train, subseries_length)
    test_data = ImageSequenceDataset(X_test, subseries_length)

    batch_size = batch_size

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    img_size = img_size
    patch_size = patch_size
    num_layers = num_layers
    num_heads = num_heads
    embed_dim = embed_dim
    forward_expansion = forward_expansion
    dropout = dropout

    model = ImageTransformer(
        img_size,
        patch_size,
        subseries_length,
        embed_dim,
        num_heads,
        num_layers,
        dropout,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if mode == "gif_generation":
        torch.save(
            model.state_dict(),
            os.path.join(model_output_dir, f"{experiment_id}_epoch_0.pth"),
        )

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    score_file = f"losses_{experiment_id}.txt"
    if not os.path.isfile(score_file):
        print("initializing score file")
        with open(score_file, "w") as f:
            f.write(
                "epoch,train_loss,val_loss,batch_size,img_size,patch_size,num_layers,num_heads,embed_dim,forward_expansion,dropout\n"
            )

    # Define the learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Number of epochs
    num_epochs = n_epochs
    best_val_loss = float("inf")
    # Training loop
    for epoch in range(num_epochs):
        start = time.time()
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            # Move data to the appropriate device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Compute average training loss
        avg_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():  # No need to track gradients during validation
            for i, (inputs, targets) in enumerate(val_loader):
                # Move data to the appropriate device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)

                if epoch % 10 == 0 and i == 0:
                    plot_current_state(
                        inputs,
                        targets,
                        model,
                        epoch,
                        device,
                        experiment_id,
                        image_output_dir,
                    )

                # Compute loss
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            print(f"Saving model with validation loss: {best_val_loss:.4f}")
            torch.save(
                model.state_dict(),
                os.path.join(model_output_dir, f"{experiment_id}_best_model.pth"),
            )

        if mode == "gif_generation":
            torch.save(
                model.state_dict(),
                os.path.join(
                    model_output_dir, f"{experiment_id}_epoch_{epoch + 1}.pth"
                ),
            )

        with open(score_file, "a") as f:
            (
                img_size,
                patch_size,
                num_layers,
                num_heads,
                embed_dim,
                forward_expansion,
                dropout,
            )
            f.write(
                f"{epoch + 1},{avg_train_loss},{avg_val_loss},{batch_size},{img_size[0]},{patch_size},{num_layers},{num_heads},{embed_dim},{forward_expansion},{dropout}\n"
            )

        current_lr = optimizer.param_groups[0]["lr"]
        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Check and print if the learning rate has changed
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            print(f"Learning rate decreased from {current_lr:.6f} to {new_lr:.6f}")
        # Print training and validation loss
        stop = time.time()
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Training Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {avg_val_loss:.4f},"
            f"Time Spent: {stop - start:.2f}"
        )
    print("Training complete!")


if __name__ == "__main__":
    run_prediction()
