import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import os
import gc
import pandas as pd
import pickle
import time

from ._utils import create_dataloader, find_ideal_learning_rate
from ._dataset import (OrganoidDataset,
                       OrganoidTrainingDataset,
                       OrganoidValidationDataset)
from .models import DenseNet121, ResNet50, MobileNetV3_Large

def run_classification_train_test(model,
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
        weight_decay = 1e-3
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_test_loss = float('inf')

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

def finetune_for_timepoints(model,
                            readout: str,
                            start_timepoint: int,
                            stop_timepoint: int,
                            learning_rate: float,
                            batch_size: int,
                            n_epochs: int,
                            experiment_id: str,
                            dataset_id: str,
                            validation_dataset_id: str,
                            output_dir = "./results",
                            model_output_dir = "./classifiers",
                            dataset_input_dir = "./raw_data") -> None:
    """\
    Trains model on timepoint preset dataset.
    Expects a pretrained model

    Learning Rate is adjusted based on valF1.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)
    
    full_dataset = OrganoidDataset.read_classification_dataset(
        os.path.join(dataset_input_dir, f"{dataset_id}.cds")
    )
    full_dataset.split_timepoints(start_timepoint, stop_timepoint)

    full_validation_dataset = OrganoidDataset.read_classification_dataset(
        os.path.join(dataset_input_dir, f"{validation_dataset_id}.cds")
    )
    full_validation_dataset.split_timepoints(start_timepoint, stop_timepoint)
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
    model.freeze_layers(-1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay = 1e-4
    )

    learning_rate = find_ideal_learning_rate(
        model = model,
        criterion = criterion,
        optimizer = optimizer,
        train_loader = train_loader,
        n_tests = 5
    )

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay = 1e-4
    )

    print(f"Ideal learning rate at {round(learning_rate, 5)}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    best_val_f1 = 0

    # Training loop
    for epoch in range(n_epochs):
        start = time.time()
        model.train()
        train_loss = 0
        train_true = []
        train_preds = []
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            target = torch.argmax(target, dim = 1)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()
            
            train_loss += loss.item()

            train_preds += torch.argmax(output, dim = 1).cpu().tolist()
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
                target = torch.argmax(target, dim = 1)

                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()

                test_preds += torch.argmax(output, dim = 1).cpu().tolist()
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
                target = torch.argmax(target, dim = 1)

                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()

                val_preds += torch.argmax(output, dim = 1).cpu().tolist()
                val_true += target.cpu().tolist()
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='weighted')

        # Get the current learning rate before scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        
        # Step the learning rate scheduler based on the validation loss
        scheduler.step(val_f1)
        
        # Check if the learning rate has been reduced
        # new_lr = optimizer.param_groups[0]['lr']
        # if new_lr < current_lr:
        #     # load best performing model before continuing
        #     model.load_state_dict(
        #         torch.load(
        #             os.path.join(
        #                 model_output_dir,
        #                 f'{model.__class__.__name__}_{readout}_base_model.pth'
        #             )
        #         )
        #     )
        #     print(f"[INFO] Learning rate reduced from {current_lr} to {new_lr}")


        stop = time.time()

        # Print metrics
        print(
            f"[INFO] Epoch: {epoch+1}/{n_epochs}, "
            f"Train loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Val loss: {val_loss:.4f}, "
            f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}, Val Accuracy: {val_acc:.4f}, "
            f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Val F1: {val_f1:.4f}, "
            f"Time: {stop-start}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(), 
                os.path.join(
                    model_output_dir,
                    f'{model.__class__.__name__}_{readout}_tp_{start_timepoint}_{stop_timepoint}.pth'
                )
            )
            print(f'Saved best model with val F1: {best_val_f1:.4f}')
        
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

def run_experiment_cross_validation(model: str,
                                    readout: str,
                                    learning_rate: float,
                                    batch_size: int,
                                    n_epochs: int,
                                    experiment_id: str,
                                    output_dir = "./results",
                                    model_output_dir = "./classifiers",
                                    dataset_input_dir = "./raw_data",
                                    calculate_learning_rate: bool = True,
                                    num_classes: int = 2) -> None:

    experiments = [
        "E001",
        "E002",
        "E004",
        "E005",
        "E006",
        "E007",
        "E008",
        "E009",
        "E010",
        "E011",
        "E012"
    ]

    for experiment in experiments:
        if model == "DenseNet121":
            _model = DenseNet121(num_classes = num_classes)
        elif model == "ResNet50":
            _model = ResNet50(num_classes = num_classes)
        elif model == "MobileNetV3_Large":
            _model = MobileNetV3_Large(num_classes = num_classes)
        else:
            raise ValueError("model not found")
        _cross_validation_train_loop(model = _model,
                                     readout = readout,
                                     learning_rate = learning_rate,
                                     batch_size = batch_size,
                                     n_epochs = n_epochs,
                                     experiment_id = experiment_id,
                                     dataset_id = f"M{experiment}_full_SL3_fixed",
                                     validation_dataset_id = f"{experiment}_full_SL3_fixed",
                                     output_dir = output_dir,
                                     model_output_dir = model_output_dir,
                                     dataset_input_dir = dataset_input_dir,
                                     calculate_learning_rate = calculate_learning_rate)

        gc.collect()
                                     




def _cross_validation_train_loop(model,
                                 readout: str,
                                 learning_rate: float,
                                 batch_size: int,
                                 n_epochs: int,
                                 experiment_id: str,
                                 dataset_id: str,
                                 validation_dataset_id: str,
                                 output_dir = "./results",
                                 model_output_dir = "./classifiers",
                                 dataset_input_dir = "./raw_data",
                                 calculate_learning_rate: bool = True,
                                 weighted_loss: bool = True) -> None:
    """\
    Trains model on full dataset in order to find a good baseline model.

    We use weighted Loss.

    Learning Rate is adjusted based on valF1.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)

    val_exp = validation_dataset_id[:4]
    
    full_dataset = OrganoidDataset.read_classification_dataset(
        os.path.join(dataset_input_dir, f"{dataset_id}.cds")
    )
    full_dataset._create_class_counts()

    full_validation_dataset = OrganoidDataset.read_classification_dataset(
        os.path.join(dataset_input_dir, f"{validation_dataset_id}.cds")
    )
    full_validation_dataset._create_class_counts()

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
    if os.path.isfile(output_file):
        results = pd.read_csv(output_file)
        results = results[results["Model"] == model.__class__.__name__].copy()
        if val_exp in results["ValExpID"].unique():
            print(f"Skipping {val_exp} as it is already calculated")
            return

    mode = "w" if not os.path.isfile(output_file) else "a"
    with open(output_file, mode) as file:
        file.write(
            "ExperimentID,ValExpID,Readout,Model,LearningRate,"
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
        batch_size = batch_size, shuffle = True, train = True,
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
    
    if weighted_loss is True:
        class_weights = 1 / (y_train.sum(axis=0) / y_train.shape[0])
        class_weights = torch.tensor(class_weights).float().to(device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight = class_weights)
    
    if calculate_learning_rate is True:
        try:
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                weight_decay = 1e-4
            )
            learning_rate = find_ideal_learning_rate(
                model = model,
                criterion = criterion,
                optimizer = optimizer,
                train_loader = train_loader,
                n_tests = 5
            )
        except Exception as e:
            print(str(e))
            print("Skipping LR calculation")
            if model.__class__.__name__ == "DenseNet121":
                learning_rate = 1e-5
            elif model.__class__.__name__ == "ResNet50":
                learning_rate = 0.0001
            elif model.__class__.__name__ == "MobileNetV3_Large":
                learning_rate = 0.0003
            else:
                learning_rate = 0.0003

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay = 1e-4
    )

    print(f"Ideal learning rate at {round(learning_rate, 5)}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    best_val_f1 = 0
    best_test_f1 = 0

    loss_dict_train = {epoch: [] for epoch in range(n_epochs)}
    loss_dict_test = {epoch: [] for epoch in range(n_epochs)}
    loss_dict_val = {epoch: [] for epoch in range(n_epochs)}


    # Training loop
    for epoch in range(n_epochs):
        start = time.time()
        model.train()
        train_loss = 0
        train_true = []
        train_preds = []
        train_loss_list = []
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            target = torch.argmax(target, dim = 1)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()
            
            train_loss += loss.item()
            train_loss_list.append(loss.item())

            train_preds += torch.argmax(output, dim = 1).cpu().tolist()
            train_true += target.cpu().tolist()

        loss_dict_train[epoch] = train_loss_list

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_true, train_preds)
        train_f1 = f1_score(train_true, train_preds, average='weighted')

        # Validation loop
        model.eval()
        test_loss = 0
        test_true = []
        test_preds = []
        test_loss_list = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                target = torch.argmax(target, dim = 1)

                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                test_loss_list.append(loss.item())

                test_preds += torch.argmax(output, dim = 1).cpu().tolist()
                test_true += target.cpu().tolist()
        
        loss_dict_test[epoch] = test_loss_list

        test_loss /= len(test_loader)
        test_acc = accuracy_score(test_true, test_preds)
        test_f1 = f1_score(test_true, test_preds, average='weighted')
        
        # Validation loop
        model.eval()
        val_loss = 0
        val_true = []
        val_preds = []
        val_loss_list = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                target = torch.argmax(target, dim = 1)

                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                val_loss_list.append(loss.item())

                val_preds += torch.argmax(output, dim = 1).cpu().tolist()
                val_true += target.cpu().tolist()

        loss_dict_val[epoch] = val_loss_list

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='weighted')

        # Step the learning rate scheduler based on the validation loss
        scheduler.step(val_f1)
        
        stop = time.time()

        # Print metrics
        print(
            f"[INFO] Epoch: {epoch+1}/{n_epochs}, "
            f"Train loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Val loss: {val_loss:.4f}, "
            f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}, Val Accuracy: {val_acc:.4f}, "
            f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Val F1: {val_f1:.4f}, "
            f"Time: {stop-start}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(), 
                os.path.join(
                    model_output_dir,
                    f'{model.__class__.__name__}_val_{val_exp}_{readout}_base_model.pth'
                )
            )
            print(f'Saved best model with val F1: {best_val_f1:.4f}')

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1 
            torch.save(
                model.state_dict(), 
                os.path.join(
                    model_output_dir,
                    f'{model.__class__.__name__}_test_{val_exp}_{readout}_base_model.pth'
                )
            )
            print(f'Saved best model with test F1: {best_val_f1:.4f}')

        torch.save(
            model.state_dict(), 
            os.path.join(
                model_output_dir,
                f'{model.__class__.__name__}_epoch_{epoch+1}_{val_exp}_{readout}.pth'
            )
        )
        print(f'Saved best model with test F1: {best_val_f1:.4f}')
    
        # Write metrics to CSV file
        with open(output_file, "a") as file:
            file.write(
                f"{experiment_id},{val_exp},{readout},"
                f"{model.__class__.__name__},{learning_rate},"
                f"{batch_size},{epoch+1},{train_loss},{test_loss},{val_loss},"
                f"{train_acc},{test_acc},{val_acc},{train_f1},{test_f1},{val_f1},"
                f"{dataset_id},{start_timepoint}-{stop_timepoint},"
                f"{bbox_cropped},{bbox_rescaling}\n"
            )

    with open(os.path.join(output_dir, f"train_losses_{val_exp}_{readout}_{model.__class__.__name__}.dict"), "wb") as file:
        pickle.dump(loss_dict_train, file)

    with open(os.path.join(output_dir, f"test_losses_{val_exp}_{readout}_{model.__class__.__name__}.dict"), "wb") as file:
        pickle.dump(loss_dict_test, file)

    with open(os.path.join(output_dir, f"val_losses_{val_exp}_{readout}_{model.__class__.__name__}.dict"), "wb") as file:
        pickle.dump(loss_dict_val, file)

    return

def find_base_model(model,
                    readout: str,
                    learning_rate: float,
                    batch_size: int,
                    n_epochs: int,
                    experiment_id: str,
                    dataset_id: str,
                    validation_dataset_id: str,
                    output_dir = "./results",
                    model_output_dir = "./classifiers",
                    dataset_input_dir = "./raw_data",
                    calculate_learning_rate: bool = True) -> None:
    """\
    Trains model on full dataset in order to find a good baseline model.

    We use weighted Loss.

    Learning Rate is adjusted based on valF1.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)
    
    full_dataset = OrganoidDataset.read_classification_dataset(
        os.path.join(dataset_input_dir, f"{dataset_id}.cds")
    )
    full_dataset._create_class_counts()

    full_validation_dataset = OrganoidDataset.read_classification_dataset(
        os.path.join(dataset_input_dir, f"{validation_dataset_id}.cds")
    )
    full_validation_dataset._create_class_counts()

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
        batch_size = batch_size, shuffle = True, train = True,
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

    class_weights = 1 / (y_train.sum(axis=0) / y_train.shape[0])
    class_weights = torch.tensor(class_weights).float().to(device)
    criterion = nn.CrossEntropyLoss(weight = class_weights)
    
    if calculate_learning_rate is True:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay = 1e-4
        )
        learning_rate = find_ideal_learning_rate(
            model = model,
            criterion = criterion,
            optimizer = optimizer,
            train_loader = train_loader,
            n_tests = 5
        )

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay = 1e-4
    )

    print(f"Ideal learning rate at {round(learning_rate, 5)}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    best_val_f1 = 0

    loss_dict_train = {epoch: [] for epoch in range(n_epochs)}
    loss_dict_test = {epoch: [] for epoch in range(n_epochs)}
    loss_dict_val = {epoch: [] for epoch in range(n_epochs)}


    # Training loop
    for epoch in range(n_epochs):
        start = time.time()
        model.train()
        train_loss = 0
        train_true = []
        train_preds = []
        train_loss_list = []
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            target = torch.argmax(target, dim = 1)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()
            
            train_loss += loss.item()
            train_loss_list.append(loss.item())

            train_preds += torch.argmax(output, dim = 1).cpu().tolist()
            train_true += target.cpu().tolist()

        loss_dict_train[epoch] = train_loss_list

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_true, train_preds)
        train_f1 = f1_score(train_true, train_preds, average='weighted')

        # Validation loop
        model.eval()
        test_loss = 0
        test_true = []
        test_preds = []
        test_loss_list = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                target = torch.argmax(target, dim = 1)

                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                test_loss_list.append(loss.item())

                test_preds += torch.argmax(output, dim = 1).cpu().tolist()
                test_true += target.cpu().tolist()
        
        loss_dict_test[epoch] = test_loss_list

        test_loss /= len(test_loader)
        test_acc = accuracy_score(test_true, test_preds)
        test_f1 = f1_score(test_true, test_preds, average='weighted')
        
        # Validation loop
        model.eval()
        val_loss = 0
        val_true = []
        val_preds = []
        val_loss_list = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                target = torch.argmax(target, dim = 1)

                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                val_loss_list.append(loss.item())

                val_preds += torch.argmax(output, dim = 1).cpu().tolist()
                val_true += target.cpu().tolist()

        loss_dict_val[epoch] = val_loss_list

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='weighted')

        # Get the current learning rate before scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        
        # Step the learning rate scheduler based on the validation loss
        scheduler.step(val_f1)
        
        # Check if the learning rate has been reduced
        # new_lr = optimizer.param_groups[0]['lr']
        # if new_lr < current_lr:
        #     # load best performing model before continuing
        #     model.load_state_dict(
        #         torch.load(
        #             os.path.join(
        #                 model_output_dir,
        #                 f'{model.__class__.__name__}_{readout}_base_model.pth'
        #             )
        #         )
        #     )
        #     print(f"[INFO] Learning rate reduced from {current_lr} to {new_lr}")


        stop = time.time()

        # Print metrics
        print(
            f"[INFO] Epoch: {epoch+1}/{n_epochs}, "
            f"Train loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Val loss: {val_loss:.4f}, "
            f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}, Val Accuracy: {val_acc:.4f}, "
            f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Val F1: {val_f1:.4f}, "
            f"Time: {stop-start}"
        )

        if True:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(), 
                os.path.join(
                    model_output_dir,
                    f'{model.__class__.__name__}_{readout}_base_model_epoch_{epoch+1}.pth'
                )
            )
            print(f'Saved best model with val F1: {best_val_f1:.4f}')
        
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

    with open(os.path.join(output_dir, "train_losses.dict"), "wb") as file:
        pickle.dump(loss_dict_train, file)

    with open(os.path.join(output_dir, "test_losses.dict"), "wb") as file:
        pickle.dump(loss_dict_test, file)

    with open(os.path.join(output_dir, "val_losses.dict"), "wb") as file:
        pickle.dump(loss_dict_val, file)

if __name__ == "__main__":
    pass
