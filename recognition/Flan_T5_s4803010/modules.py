"""
COMP3710 - Pattern Recognition - modules.py

Author: Nicholas Ardika
Student ID: 48030102

This file ... TODO
"""

import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

class FLANT5(nn.Module):
    def __init__(self, modelName="google/flan-t5-base"):
        super().__init__()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(modelName)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)