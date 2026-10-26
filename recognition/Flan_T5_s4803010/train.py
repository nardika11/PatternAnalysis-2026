"""
COMP3710 - Pattern Recognition - train.py

Author: Nicholas Ardika
Student ID: 48030102

This file ... TODO
"""

import torch

def train_single_epoch(model, trainLoader, optimiser, scheduler, scaler):
    model.train()
    device = next(model.parameters()).device

    totalLoss = 0.0
    for batch in trainLoader:
        batch = {key: value.to(device) for key, value in batch.items()}

        optimiser.zero_grad()

        # Utilises mixed precision
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            # Passes the input ids, attention masks and labels to FLAN-T5 for a forward pass
            output = model(**batch)

            # FLAN-T5 automatically produces loss, so specifying loss function isn't needed
            loss = output.loss

        # Backprop with scaled losses due to mixed precision
        scaler.scale(loss).backward()

        # Apply gradient clipping to limit gradients exploding 
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        totalLoss += loss.item()

    return totalLoss / len(trainLoader)

def validate(model, valLoader):
    model.eval()
    device = next(model.parameters()).device
    
    totalLoss = 0.0
    with torch.no_grad():
        for batch in valLoader:
            batch = {key: value.to(device) for key, value in batch.items()}

            # Utilises mixed precision same as training
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                output = model(**batch)
                loss = output.loss

            totalLoss += loss.item()

    return totalLoss / len(valLoader)