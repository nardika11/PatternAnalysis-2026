"""
COMP3710 - Pattern Recognition - train.py

Author: Nicholas Ardika
Student ID: 48030102

This file ... TODO
"""

import torch
import random
import evaluate
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup
from dataset import BioLaySummData
from modules import FLANT5

ROOT_PATH = "./recognition/Flan_T5_s4803010"

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

def train(model, trainLoader, valLoader, epochs, lr, patience):
    device = next(model.parameters()).device

    startTime = time()
    trainableParams = [param for param in model.parameters() if param.requires_grad]

    optimiser = torch.optim.AdamW(trainableParams, lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimiser,
        num_training_steps=len(trainLoader) * epochs,
        num_warmup_steps=int(len(trainLoader) * epochs * 0.05))

    scaler = torch.amp.GradScaler(device.type)

    trainLosses, valLosses = [], []
    bestValLoss = float("inf")
    epochsWithoutImproving = 0
    bestModelPath = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    trainProgress = tqdm(range(epochs), desc="Training", unit="epoch", ncols=100)
    for epoch in trainProgress:
        trainLoss = train_single_epoch(model, trainLoader, optimiser, scheduler, scaler)
        valLoss = validate(model, valLoader)

        trainLosses.append(trainLoss)
        valLosses.append(valLoss)

        trainProgress.set_postfix({"Train Loss": f"{trainLoss:.4f}",
                                   "Val Loss": f"{valLoss:.4f}"})

        if valLoss < bestValLoss:
            bestValLoss = valLoss
            epochsWithoutImproving = 0

            trainingType = "LoRA" if model.useLora else "Full"
            bestModelPath = f"{ROOT_PATH}/models/FlanT5_{trainingType}_lr{lr}"
            model.model.save_pretrained(bestModelPath)
        else:
            epochsWithoutImproving += 1

        if epochsWithoutImproving >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    trainingTime = time() - startTime

    if device.type == "cuda":
        maxVram = torch.cuda.max_memory_allocated()
    else:
        maxVram = 0.0

    if bestModelPath is not None:
        model.load_tuned_model(bestModelPath)
    
    return {"Train Losses": trainLosses,
            "Val Losses": valLosses,
            "Training Time": trainingTime,
            "Max VRAM": maxVram}

def evaluate_rouge(model, valLoader, tokeniser):
    model.eval()
    device = next(model.parameters()).device

    generatedSummaries = []
    originalSummaries = []
    totalGenerationTime = 0.0
    totalTokensGenerated = 0.0

    with torch.no_grad():
        for batch in tqdm(valLoader, desc="Evaluating rouge", unit="batch", ncols=100):
            batch = {key: value.to(device) for key, value in batch.items()}

            startTime = time()
            batchOutput = model.generate(input_ids=batch["input_ids"], 
                                         attention_mask=batch["attention_mask"],
                                         max_new_tokens=512)
            totalGenerationTime += time() - startTime

            specialTokens = set(tokeniser.all_special_ids)

            for seq in batchOutput:
                totalTokensGenerated += sum([token.item() not in specialTokens
                                             for token in seq])

            outputSummaries = tokeniser.batch_decode(batchOutput, skip_special_tokens=True)

            labels = batch["labels"].clone()
            labels[labels == -100] = tokeniser.pad_token_id
            labelSummaries = tokeniser.batch_decode(labels, skip_special_tokens=True)

            generatedSummaries.extend(outputSummaries)
            originalSummaries.extend(labelSummaries)

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=generatedSummaries, 
                           references=originalSummaries, 
                           use_stemmer=True)

    inferenceThroughput = totalTokensGenerated / totalGenerationTime if totalGenerationTime > 0 else 0.0

    return {**scores, "Inference Throughput": inferenceThroughput}

def test(model, testLoader, tokeniser, seed=12, numCases=5):
    model.eval()
    device = next(model.parameters()).device

    originalReports = []
    generatedSummaries = []

    rng = random.Random(seed)
    indices = rng.sample(range(len(testLoader.dataset)), numCases)

    testSubset = DataLoader(Subset(testLoader.dataset, indices), 
                           batch_size=1, shuffle=False,
                           collate_fn=testLoader.collate_fn)

    with torch.no_grad():
        for batch in testSubset:
            batch = {key: value.to(device) for key, value in batch.items()}

            batchOutput = model.generate(input_ids=batch["input_ids"], 
                                         attention_mask=batch["attention_mask"],
                                         max_new_tokens=512)

            originalReport = tokeniser.decode(batch["input_ids"][0], skip_special_tokens=True)
            originalReport = originalReport.removeprefix("Summarise the medical report into lay language for a general audience: ").strip()

            outputSummary = tokeniser.decode(batchOutput[0], skip_special_tokens=True)

            originalReports.append(originalReport)
            generatedSummaries.append(outputSummary)

    print("Manual Test Evaluation!")
    for i in range(numCases):
        print(f"\nOriginal Report: {originalReports[i]}")
        print(f"Generated Summary: {generatedSummaries[i]}")

    return {"Original Reports": originalReports, 
            "Generated Summaries": generatedSummaries}