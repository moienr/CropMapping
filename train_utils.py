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