"""
COMP3710 - Pattern Recognition - modules.py

Author: Nicholas Ardika
Student ID: 48030102

This file ... TODO
"""

import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model

class FLANT5(nn.Module):
    def __init__(self, modelName="google/flan-t5-base", useLora=False):
        super().__init__()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(modelName)

        if useLora:
            self.apply_lora()

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def apply_lora(self):
        loraConfig = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, 
                                target_modules="all-linear", 
                                bias="none", task_type="SEQ_2_SEQ_LM")

        self.model = get_peft_model(self.model, loraConfig)

    def display_parameter_counts(self):
        total = sum(param.numel() for param in self.model.parameters())
        trainable = sum(param.numel() for param in self.model.parameters() if param.requires_grad)

        print(f"Total parameters: {total} ~ Trainable parameters: {trainable}")

if __name__ == "__main__":
    model = FLANT5(useLora=True)
    model.display_parameter_counts()

    model = FLANT5()
    model.display_parameter_counts()