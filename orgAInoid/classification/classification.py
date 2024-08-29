import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import os
import time

from ._utils import create_dataloader
from ._dataset import read_classification_dataset

def run_classification(model,
                       learning_rate: float,
                       batch_size: int,
                       n_epochs: int,
                       experiment_id: str,
                       dataset_id: str,
                       output_dir = "./results",
                       model_output_dir = "./classifiers",
                       dataset_input_dir = "./raw_data") -> None:

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)
    
    dataset = read_classification_dataset(
        os.path.join(dataset_input_dir, f"{dataset_id}.cds")
    )

    start_timepoint = dataset.start_timepoint
    stop_timepoint = dataset.stop_timepoint
    bbox_cropped = dataset.cropped_bbox
    bbox_rescaling = dataset.rescale_cropped_image
    if hasattr(dataset, "readout"):
        classified_variable = dataset.readout
    else:
        classified_variable = "RPE"
 
    X_train, y_train, X_test, y_test = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
    assert X_train is not None
    assert y_train is not None
    assert X_test is not None
    assert y_test is not None

    learning_rate = learning_rate or 0.0001
    batch_size = batch_size or 64
    n_epochs = n_epochs or 200

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_file = os.path.join(output_dir, f"{experiment_id}.txt")
    mode = "w" if not os.path.isfile(output_file) else "a"
    with open(output_file, mode) as file:
        file.write(
            "ExperimentID,Readout,Model,LearningRate,"
            "BatchSize,Epoch,TrainLoss,ValLoss,"
            "TrainAccuracy,ValAccuracy,TrainF1,ValF1,"
            "DatasetID,TimePoints,CroppedBbox,RescaledBBox\n"
        )


    # Print current configuration
    print(f"[INFO] Starting experiment with Model: {model.__class__.__name__}, "
          f"Learning Rate: {learning_rate}, Batch Size: {batch_size}")

    # Initialize the dataloaders
    train_loader = create_dataloader(X_train, y_train, batch_size = batch_size, shuffle = True, train = True)
    val_loader = create_dataloader(X_test, y_test, batch_size = batch_size, shuffle = False, train = False)

    # Initialize the model, criterion, and optimizer
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(n_epochs):
        start = time.time()
        model.train()
        train_loss = 0
        train_true = []
        train_preds = []
        
        for data, target in train_loader:
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
            for data, target in val_loader:
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
            # load best performing model before continuing
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        model_output_dir,
                        f'{model.__class__.__name__}_{classified_variable}.pth'
                    )
                )
            )
            print(f"[INFO] Learning rate reduced from {current_lr} to {new_lr}")


        stop = time.time()

        # Print metrics
        print(
            f"[INFO] Epoch: {epoch+1}/{n_epochs}, "
            f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
            f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, "
            f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, "
            f"Time: {stop-start}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), 
                os.path.join(
                    model_output_dir,
                    f'{model.__class__.__name__}_{classified_variable}.pth'
                )
            )
            print(f'Saved best model with val loss: {best_val_loss:.4f}')
        
        # Write metrics to CSV file
        with open(output_file, "a") as file:
            file.write(
                f"{experiment_id},{classified_variable},"
                f"{model.__class__.__name__},{learning_rate},"
                f"{batch_size},{epoch+1},{train_loss},{val_loss},"
                f"{train_acc},{val_acc},{train_f1},{val_f1},"
                f"{dataset_id},{start_timepoint}-{stop_timepoint},"
                f"{bbox_cropped},{bbox_rescaling}\n"
            )


if __name__ == "__main__":
    pass
