import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import os
import time

from ._utils import create_dataloader
from ._dataset import (OrganoidDataset,
                       OrganoidTrainingDataset,
                       OrganoidValidationDataset)


def learning_rate_batch_size_test(model,
                                  readout: str,
                                  learning_rate: float,
                                  batch_size: int,
                                  n_epochs: int,
                                  experiment_id: str,
                                  dataset_id: str,
                                  validation_dataset_id: str,
                                  output_dir = "./results",
                                  model_output_dir = "./classifiers",
                                  dataset_input_dir = "./raw_data") -> None:
    """
    This test is meant to check how freezing the layers affects the overfit
    The model layers will be frozen after the first epoch, except for the
    final fully connected layer.
    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)
    
    full_dataset = OrganoidDataset.read_classification_dataset(
        os.path.join(dataset_input_dir, f"{dataset_id}.cds")
    )

    full_validation_dataset = OrganoidDataset.read_classification_dataset(
        os.path.join(dataset_input_dir, f"{validation_dataset_id}.cds")
    )
    validation_set = OrganoidValidationDataset(
        full_validation_dataset,
        readout = readout
    )

    start_timepoint = full_dataset.dataset_metadata.start_timepoint
    stop_timepoint = full_dataset.dataset_metadata.stop_timepoint
    bbox_cropped = full_dataset.image_metadata.cropped_bbox
    bbox_rescaling = full_dataset.image_metadata.rescale_cropped_image

    dataset = OrganoidTrainingDataset(
        full_dataset,
        readout = readout,
        test_size = 0.1
    )
 
    X_train, y_train, X_test, y_test = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
    X_val = validation_set.X
    y_val = validation_set.y
    assert X_train is not None
    assert y_train is not None
    assert X_test is not None
    assert y_test is not None

    # learning_rate = learning_rate or 0.0001
    # batch_size = batch_size or 64
    n_epochs = n_epochs or 200

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_file = os.path.join(output_dir, f"{experiment_id}.txt")
    mode = "w" if not os.path.isfile(output_file) else "a"
    with open(output_file, mode) as file:
        file.write(
            "ExperimentID,Readout,Model,LearningRate,"
            "BatchSize,Epoch,TrainLoss,TestLoss,ValLoss,"
            "TrainAccuracy,TestAccuracy,ValAccuracy,TrainF1,TestF1,ValF1,"
            "DatasetID,TimePoints,CroppedBbox,RescaledBBox\n"
        )


    # Print current configuration
    print(f"[INFO] Starting experiment with Model: {model.__class__.__name__}, "
          f"Learning Rate: {learning_rate}, Batch Size: {batch_size}")

    # Initialize the dataloaders
    train_loader = create_dataloader(
        X_train, y_train,
        batch_size = batch_size, shuffle = True, train = True
    )
    test_loader = create_dataloader(
        X_test, y_test,
        batch_size = batch_size, shuffle = False, train = False
    )
    val_loader = create_dataloader(
        X_val, y_val,
        batch_size = batch_size, shuffle = False, train = False
    )

    # Initialize the model, criterion, and optimizer
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    best_test_loss = float('inf')

    # Training loop
    for learning_rate in [1e-3, 1e-4, 1e-5]:
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay = 1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        for batch_size in [32, 64, 128]:
            for epoch in range(n_epochs):
                if epoch > 0:
                    model.freeze_layers(-1)
                    
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
                train_f1 = f1_score(train_true, train_preds, average='weighted')

                # Validation loop
                model.eval()
                test_loss = 0
                test_true = []
                test_preds = []
                
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        loss = criterion(output.squeeze(), target.float())
                        
                        test_loss += loss.item()

                        test_preds += torch.round(torch.sigmoid(output)).cpu().tolist()
                        test_true += target.cpu().tolist()
                
                test_loss /= len(test_loader)
                test_acc = accuracy_score(test_true, test_preds)
                test_f1 = f1_score(test_true, test_preds, average='weighted')
                
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
                val_f1 = f1_score(val_true, val_preds, average='weighted')

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
                                f'{model.__class__.__name__}_{readout}.pth'
                            )
                        )
                    )
                    print(f"[INFO] Learning rate reduced from {current_lr} to {new_lr}")


                stop = time.time()

                # Print metrics
                print(
                    f"[INFO] Epoch: {epoch+1}/{n_epochs}, "
                    f"Train loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Val loss: {val_loss:.4f}, "
                    f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}, Val Accuracy: {val_acc:.4f}, "
                    f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Val F1: {val_f1:.4f}, "
                    f"Time: {stop-start}"
                )

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    torch.save(
                        model.state_dict(), 
                        os.path.join(
                            model_output_dir,
                            f'{model.__class__.__name__}_{readout}.pth'
                        )
                    )
                    print(f'Saved best model with test loss: {best_test_loss:.4f}')
                
                # Write metrics to CSV file
                with open(output_file, "a") as file:
                    file.write(
                        f"{experiment_id},{readout},"
                        f"{model.__class__.__name__},{learning_rate},"
                        f"{batch_size},{epoch+1},{train_loss},{test_loss},{val_loss},"
                        f"{train_acc},{test_acc},{val_acc},{train_f1},{test_f1},{val_f1},"
                        f"{dataset_id},{start_timepoint}-{stop_timepoint},"
                        f"{bbox_cropped},{bbox_rescaling}\n"
                    )


def freezing_test(model,
                  readout: str,
                  learning_rate: float,
                  batch_size: int,
                  n_epochs: int,
                  experiment_id: str,
                  dataset_id: str,
                  validation_dataset_id: str,
                  output_dir = "./results",
                  model_output_dir = "./classifiers",
                  dataset_input_dir = "./raw_data") -> None:
    """
    This test is meant to check how freezing the layers affects the overfit
    The model layers will be frozen after the first epoch, except for the
    final fully connected layer.
    """

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)
    
    full_dataset = OrganoidDataset.read_classification_dataset(
        os.path.join(dataset_input_dir, f"{dataset_id}.cds")
    )

    full_validation_dataset = OrganoidDataset.read_classification_dataset(
        os.path.join(dataset_input_dir, f"{validation_dataset_id}.cds")
    )
    validation_set = OrganoidValidationDataset(
        full_validation_dataset,
        readout = readout
    )

    start_timepoint = full_dataset.dataset_metadata.start_timepoint
    stop_timepoint = full_dataset.dataset_metadata.stop_timepoint
    bbox_cropped = full_dataset.image_metadata.cropped_bbox
    bbox_rescaling = full_dataset.image_metadata.rescale_cropped_image

    dataset = OrganoidTrainingDataset(
        full_dataset,
        readout = readout,
        test_size = 0.1
    )
 
    X_train, y_train, X_test, y_test = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
    X_val = validation_set.X
    y_val = validation_set.y
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
            "BatchSize,Epoch,TrainLoss,TestLoss,ValLoss,"
            "TrainAccuracy,TestAccuracy,ValAccuracy,TrainF1,TestF1,ValF1,"
            "DatasetID,TimePoints,CroppedBbox,RescaledBBox\n"
        )


    # Print current configuration
    print(f"[INFO] Starting experiment with Model: {model.__class__.__name__}, "
          f"Learning Rate: {learning_rate}, Batch Size: {batch_size}")

    # Initialize the dataloaders
    train_loader = create_dataloader(
        X_train, y_train,
        batch_size = batch_size, shuffle = True, train = True
    )
    test_loader = create_dataloader(
        X_test, y_test,
        batch_size = batch_size, shuffle = False, train = False
    )
    val_loader = create_dataloader(
        X_val, y_val,
        batch_size = batch_size, shuffle = False, train = False
    )

    # Initialize the model, criterion, and optimizer
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay = 1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_test_loss = float('inf')

    # Training loop
    for epoch in range(n_epochs):
        if epoch > 0:
            model.freeze_layers(-1)
            
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
        train_f1 = f1_score(train_true, train_preds, average='weighted')

        # Validation loop
        model.eval()
        test_loss = 0
        test_true = []
        test_preds = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output.squeeze(), target.float())
                
                test_loss += loss.item()

                test_preds += torch.round(torch.sigmoid(output)).cpu().tolist()
                test_true += target.cpu().tolist()
        
        test_loss /= len(test_loader)
        test_acc = accuracy_score(test_true, test_preds)
        test_f1 = f1_score(test_true, test_preds, average='weighted')
        
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
        val_f1 = f1_score(val_true, val_preds, average='weighted')

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
                        f'{model.__class__.__name__}_{readout}.pth'
                    )
                )
            )
            print(f"[INFO] Learning rate reduced from {current_lr} to {new_lr}")


        stop = time.time()

        # Print metrics
        print(
            f"[INFO] Epoch: {epoch+1}/{n_epochs}, "
            f"Train loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Val loss: {val_loss:.4f}, "
            f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}, Val Accuracy: {val_acc:.4f}, "
            f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Val F1: {val_f1:.4f}, "
            f"Time: {stop-start}"
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(
                model.state_dict(), 
                os.path.join(
                    model_output_dir,
                    f'{model.__class__.__name__}_{readout}.pth'
                )
            )
            print(f'Saved best model with test loss: {best_test_loss:.4f}')
        
        # Write metrics to CSV file
        with open(output_file, "a") as file:
            file.write(
                f"{experiment_id},{readout},"
                f"{model.__class__.__name__},{learning_rate},"
                f"{batch_size},{epoch+1},{train_loss},{test_loss},{val_loss},"
                f"{train_acc},{test_acc},{val_acc},{train_f1},{test_f1},{val_f1},"
                f"{dataset_id},{start_timepoint}-{stop_timepoint},"
                f"{bbox_cropped},{bbox_rescaling}\n"
            )


def hyperparameter_model_search(experiment_id: str,
                                dataset_id: str,
                                models: list,
                                output_dir = "./results",
                                dataset_input_dir = "./raw_data",
                                test_mode: bool = False) -> None:

    raise NotImplementedError("This function contains errors")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    dataset = read_classification_dataset(
        os.path.join(dataset_input_dir, f"{dataset_id}.cds")
    )
    start_timepoint = dataset.start_timepoint
    stop_timepoint = dataset.stop_timepoint
    bbox_cropped = dataset.cropped_bbox
    bbox_rescaling = dataset.rescale_cropped_image

    X_train, y_train, X_test, y_test = dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test
    assert X_train is not None
    assert y_train is not None
    assert X_test is not None
    assert y_test is not None
    
    # Learning rates to compare
    learning_rates = [0.01, 0.003, 0.001, 0.0003]

    batch_sizes = [32, 64, 128]
    
    # Number of epochs
    num_epochs = 100

    if test_mode is True:
        learning_rates = [0.001]
        batch_sizes = [64]
        num_epochs = 10

    
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Initialize the CSV file with headers
    output_file = os.path.join(output_dir, f"{experiment_id}.txt")
    mode = "w" if not os.path.isfile(output_file) else "a"
    with open(output_file, mode) as file:
        file.write(
            "ExperimentID,Model,LearningRate,"
            "BatchSize,Epoch,TrainLoss,ValLoss,"
            "TrainAccuracy,ValAccuracy,TrainF1,ValF1,"
            "DatasetID,TimePoints,CroppedBbox,RescaledBBox\n"
        )
    
    # Main loop to iterate over models, learning rates, and transformations
    for model_class in models:
        model_start = time.time()
        for lr in learning_rates:
            for batch_size in batch_sizes:

                # Print current configuration
                print(f"[INFO] Starting experiment with Model: {model_class.__name__}, "
                      f"Learning Rate: {lr}, Batch Size: {batch_size}")
           
                # Initialize the dataloaders
                train_loader = create_dataloader(X_train, y_train, batch_size = batch_size, shuffle = True, train = True)
                val_loader = create_dataloader(X_test, y_test, batch_size = batch_size, shuffle = False, train = False)
                
                # Initialize the model, criterion, and optimizer
                model = model_class().to(device)
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
                        print(f"[INFO] Learning rate reduced from {current_lr} to {new_lr}")
            
                    stop = time.time()

                    # Print metrics
                    print(
                        f"[INFO] Epoch: {epoch+1}/{num_epochs}, "
                        f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                        f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, "
                        f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, "
                        f"Time: {stop-start}"
                    )
                    
                    # Write metrics to CSV file
                    with open(output_file, "a") as file:
                        file.write(
                            f"{experiment_id},{model_class.__name__},{lr},"
                            f"{batch_size},{epoch+1},{train_loss},{val_loss},"
                            f"{train_acc},{val_acc},{train_f1},{val_f1},"
                            f"{dataset_id},{start_timepoint}-{stop_timepoint},"
                            f"{bbox_cropped},{bbox_rescaling}\n"
                        )

        model_stop = time.time()
        print("\nModel training took: ", model_stop - model_start, " seconds.\n")
