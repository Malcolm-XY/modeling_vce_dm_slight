# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 00:29:44 2024

@author: 18307
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, f1_score

def train_model(model, train_loader, device, optimizer, criterion, epochs=30):
    """
    Train a PyTorch model.

    Parameters:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        device (torch.device): Device to use for training.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (torch.nn.Module): Loss function.
        epochs (int, optional): Number of training epochs. Default is 30.

    Returns:
        torch.nn.Module: The trained model.
    """
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = (batch_data.to(device), batch_labels.to(device))

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    return model

def test_model(model, val_loader, device, criterion):
    """
    Test a PyTorch model and calculate accuracy.

    Parameters:
        model (torch.nn.Module): The trained model to evaluate.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to use for testing.
        criterion (torch.nn.Module): Loss function.

    Returns:
        dict: A dictionary containing validation metrics (accuracy, loss, recall, f1_score).
    """
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data, batch_labels = (
                batch_data.to(device),
                batch_labels.to(device),
            )

            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            correct_predictions += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

            all_labels.extend(batch_labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct_predictions / total_samples
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Validation Accuracy: {accuracy:.2f}%, Loss: {total_loss / len(val_loader):.4f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}\n')

    return {
        'accuracy': accuracy,
        'loss': total_loss / len(val_loader),
        'recall': recall,
        'f1_score': f1
    }

def cnn_validation(model, X, y, partition_ratio=0.7, partitioning='sequential',
                   batch_size=128, epochs=30, learning_rate=0.0005):
    """
    Perform cross-validation for a CNN model.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        X (array-like or torch.Tensor): 2D input data: samples x m x n or samples x channels x m x n
        y (array-like or torch.Tensor): Labels.
        partition_ratio (float, optional): Ratio of training to testing set. Default is 0.7.
        partitioning (str, optional): Type of partitioning ('sequential' or 'randomized'). Default is 'sequential'.
        batch_size (int, optional): Batch size for DataLoader. Default is 128.
        epochs (int, optional): Number of training epochs. Default is 30.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.0005.

    Returns:
        dict: A dictionary containing validation metrics (accuracy, loss, recall, f1_score).
    """
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running Device: {device}\n')

    if partitioning == 'sequential':
        total_len = len(X_tensor)
        training_len = int(total_len * partition_ratio)
        testing_len = total_len - training_len

        training_indices = list(range(training_len))
        testing_indices = list(range(training_len, total_len))

    elif partitioning == 'randomized':
        indices = torch.randperm(len(X_tensor))
        training_len = int(len(X_tensor) * partition_ratio)
        training_indices = indices[:training_len].tolist()
        testing_indices = indices[training_len:].tolist()

    else:
        raise ValueError("Invalid partitioning method. Use 'sequential' or 'randomized'.")

    train_dataset = TensorDataset(X_tensor[training_indices], y_tensor[training_indices])
    test_dataset = TensorDataset(X_tensor[testing_indices], y_tensor[testing_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    model = train_model(model, train_loader, device, optimizer, criterion, epochs=epochs)
    metrics = test_model(model, val_loader, device, criterion)

    print(f'Validation Results: {metrics}\n')
    return metrics

def cnn_cross_validation(model, X, y, folds=5, batch_size=128, epochs=30, learning_rate=0.0005):
    """
    Perform k-fold cross-validation on a PyTorch model.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        X (array-like or torch.Tensor): Input data: samples x m x n
        y (array-like or torch.Tensor): Labels.
        folds (int, optional): Number of folds for cross-validation. Default is 5.
        batch_size (int, optional): Batch size for DataLoader. Default is 128.
        epochs (int, optional): Number of training epochs. Default is 30.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 0.0005.

    Returns:
        dict: A dictionary containing average metrics across all folds.
    """
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running Device: {device}\n')

    results = []

    total_len = len(X_tensor)
    fold_size = total_len // folds
    indices = list(range(total_len))

    for fold in range(folds):
        print(f"Fold {fold + 1}/{folds}")
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < folds - 1 else total_len
        val_idx = indices[val_start:val_end]
        train_idx = indices[:val_start] + indices[val_end:]

        train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        print(f"Training Fold {fold + 1}...")
        model = train_model(model, train_loader, device, optimizer, criterion, epochs=epochs)
        metrics = test_model(model, val_loader, device, criterion)
        results.append(metrics)

    avg_metrics = {
        'accuracy': sum(r['accuracy'] for r in results) / folds,
        'loss': sum(r['loss'] for r in results) / folds,
        'recall': sum(r['recall'] for r in results) / folds,
        'f1_score': sum(r['f1_score'] for r in results) / folds
    }

    print(f"Average Metrics Across {folds} Folds: {avg_metrics}\n")
    return avg_metrics
