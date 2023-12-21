import torch
import os
from dataset.data_loaders import *
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
from config import DEVICE

def train_step(model, train_loader, criterion, optimizer, epoch, verbose=True):
    """
    Train the model for one epoch
    
    Args:
    - model: the model to train
    - train_loader: the data loader for the training data
    - criterion: the loss function
    - optimizer: the optimizer
    - epoch: the current epoch
    
    Returns:
    - None
    """
    # Set the model to train mode
    model.train()
    train_loss = 0
    num_batches = len(train_loader)
    # Loop over the data in the train loader
    for batch_idx, (s1, s2, crop_map) in enumerate(train_loader):

        # Move the data to the device
        s1, s2, crop_map = s1.to(DEVICE), s2.to(DEVICE), crop_map.to(DEVICE)
        # print(f"s1.shape: {s1.shape}", f"s2.shape: {s2.shape}", f"crop_map.shape: {crop_map.shape}")
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(s1, s2)

        # Calculate the loss
        loss = criterion(outputs, crop_map)
        train_loss += loss.item()

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        if verbose:
            # Print the loss
            print(f'Train Epoch: {epoch} [{batch_idx * len(s1)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
    train_loss /= num_batches
    return train_loss


def valid_step(model, valid_loader, criterion):
    """
    Evaluate the model on the validation set
    
    Args:
    - model: the model to evaluate
    - valid_loader: the data loader for the validation data
    - criterion: the loss function
    
    Returns:
    - val_loss: the average validation loss
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize the loss and number of samples
    val_loss = 0.0
    num_batches = len(valid_loader)

    # Disable gradient computation
    with torch.no_grad():
        # Loop over the data in the validation loader
        for s1, s2, crop_map in valid_loader:
            # Move the data to the device
            s1, s2, crop_map = s1.to(DEVICE), s2.to(DEVICE), crop_map.to(DEVICE)

            # Forward pass
            outputs = model(s1, s2)

            # Calculate the loss
            loss = criterion(outputs, crop_map)

            # Update the loss and number of samples
            val_loss += loss.item() 


    # Calculate the average validation loss
    val_loss /= num_batches

    return val_loss





def train(model, train_loader, valid_loader, criterion, optimizer, scheduler_type, num_epochs, **kwargs):
    """
    Train the model with a learning rate scheduler
    
    Args:
    - model: the model to train
    - train_loader: the data loader for the training data
    - valid_loader: the data loader for the validation data
    - criterion: the loss function
    - optimizer: the optimizer
    - scheduler_type: the type of learning rate scheduler ("constant", "step", "plateau")
    - num_epochs: the number of epochs to train for
    
    Returns:
    - results: a dictionary containing training and validation loss histories
    """
    results = {
        "train_loss_history": [],
        "val_loss_history": []
    }
    
    progress_bar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    
    if scheduler_type == "step":
        step_size = kwargs["step_size"]
        gamma = kwargs["gamma"]
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "plateau":
        factor = kwargs["factor"]
        patience = kwargs["patience"]
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    else:
        scheduler = None
    
    for epoch in progress_bar:
        train_loss = train_step(model, train_loader, criterion, optimizer, epoch + 1, verbose=False)
        val_loss = valid_step(model, valid_loader, criterion)
        
        results["train_loss_history"].append(train_loss)
        results["val_loss_history"].append(val_loss)
        
        progress_bar.set_postfix({"Epoch": epoch + 1, "Train Loss": train_loss, "Validation Loss": val_loss})
        
        # Step the learning rate scheduler
        if scheduler == "step":
            scheduler.step()
        elif scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            pass
        
    return results


def binary_mask_accuracy(predicted, true, threshold=0.3, channel=0):
    """ Calculates the metrics, for a single image, given the predicted and true binary masks.
    Parameters:
        - predicted: torch.Tensor with shape (num_channels, height, width)
        - true: torch.Tensor with shape (num_channels, height, width)
        - threshold: float, the threshold to apply to the predicted mask
        - channel: int, the channel to select from the predicted and true mask (default: 0, useful for multiclass masks)
    Returns:
        - acc_dict: dict, a dictionary with the accuracy, true positive rate (recall), true negative rate, precision and f1 score
    """
    predicted = predicted[channel, :, :]  # select the channel
    true = true[channel, :, :]  # select the channel
    predicted = predicted > threshold  # apply a threshold to the predicted mask
    true_positive = ((predicted == 1) & (true == 1)).sum().item()  # count the number of true positives
    true_negative = ((predicted == 0) & (true == 0)).sum().item()  # count the number of true negatives
    false_positive = ((predicted == 1) & (true == 0)).sum().item()  # count the number of false positives
    false_negative = ((predicted == 0) & (true == 1)).sum().item()  # count the number of false negatives
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)  # calculate the accuracy

    recall = true_positive / (true_positive + false_negative)  if (true_positive + false_negative) > 0 else 0 # calculate the recall
    true_negative_rate = true_negative / (true_negative + false_positive)  if (true_negative + false_positive) > 0 else 0 # calculate the true negative rate
    precision = true_positive / (true_positive + false_positive)  if (true_positive + false_positive) > 0 else 0 # calculate the precision
    f1_score = 2 * precision * recall / (precision + recall)  if (precision + recall) > 0 else 0 # calculate the f1 score
    acc_dict = {"accuracy": accuracy,
                "recall (true_positive_rate)": recall,
                "true_negative_rate": true_negative_rate,
                "precision": precision,
                "f1_score": f1_score}
    return acc_dict


def calculate_dataset_metrics(data_loaders, model, threshold=0.4, channel= 0):
    """
    Calculates various metrics for a given dataset using a trained model.

    Parameters:
    - data_loaders (dict): A dictionary containing the data loaders for different datasets.
    - model: The trained model to evaluate.
    - threshold (float): The threshold value for binarizing the model's output.
    - channel (int): The channel index to consider for evaluation.

    Returns:
    - metrics (dict): A dictionary containing the calculated metrics for each dataset.
      The metrics include accuracy, true positive rate, true negative rate, precision, recall, and F1 score.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    metrics = {}
    
    with torch.no_grad():
        for name, loader in data_loaders.items():
            correct = 0
            total = 0
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0
            
            for k, batch in enumerate(loader):
                print(f"Batch {k + 1} / {len(loader)}", end="\r")
                s1_img = batch[0].to(device)
                s2_img = batch[1].to(device)
                crop_map = batch[2].to(device)
                output = model(s1_img, s2_img)
                
                crop_map = crop_map[:, channel, :, :].cpu().detach()
                output = output[:, channel, :, :].cpu().detach()

                output[output >= threshold] = 1
                output[output < threshold] = 0
                
                correct += (output == crop_map).sum().item()
                total += crop_map.numel()
                true_positive += ((output == 1) & (crop_map == 1)).sum().item()
                true_negative += ((output == 0) & (crop_map == 0)).sum().item()
                false_positive += ((output == 1) & (crop_map == 0)).sum().item()
                false_negative += ((output == 0) & (crop_map == 1)).sum().item()
            
            accuracy = (correct / total) if total > 0 else 0

            recall = (true_positive / (true_positive + false_negative)) if (true_positive + false_negative) > 0 else 0
            true_negative_rate = (true_negative / (true_negative + false_positive)) if (true_negative + false_positive) > 0 else 0
            precision = (true_positive / (true_positive + false_positive)) if (true_positive + false_positive) > 0 else 0
            f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            
            metrics[name] = {
                "accuracy": accuracy,
                "true_positive_rate (recall)": recall,
                "true_negative_rate": true_negative_rate,
                "precision": precision,
                "f1_score": f1_score
            }
    
    model.train()
    return metrics
