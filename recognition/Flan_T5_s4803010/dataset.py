import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

class BioLaySummData():
    """Data class used to perform loading and preprocessing of the BioLaySumm dataset from Hugging Face"""
    def __init__(self):
        self.dataPath = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
        self.data = None

        # The maximumm length of each sequences tokenisation representation before truncation is performed
        self.maxInputTokens = 512
        self.maxLabelTokens = 512

        # LLM instruction that are added to the input texts
        self.prefix = "Summarise the medical report into lay language for a general audience: "

        self.tokeniser = None
        self.collator = None

    def load_data(self):
        self.data = load_dataset(self.dataPath)

    def load_tokeniser(self, tokeniserName="google/flan-t5-base"):
        self.tokeniser = AutoTokenizer.from_pretrained(tokeniserName)

    def load_collator(self):
        self.collator = DataCollatorForSeq2Seq(tokenizer=self.tokeniser, padding=True, label_pad_token_id=-100)

    def setup_default(self):
        self.load_tokeniser()
        self.load_collator()
        self.load_data()

    def get_collator(self):
        return self.collator

    def apply_prefix(self, data):
        return [self.prefix + report for report in data["radiology_report"]]

    def tokenise_data(self, data):
        inputs = self.apply_prefix(data)
        inputs = self.tokeniser(inputs, truncation=True, max_length=self.maxInputTokens)

        labels = self.tokeniser(text_target=data["layman_report"], truncation=True, max_length=self.maxLabelTokens)
        inputs["labels"] = labels["input_ids"]

        return inputs

    def process_split(self, split, batchSize):
        data = self.data.get(split)

        data = data.map(self.tokenise_data, batched=True, batch_size=batchSize, remove_columns=data.column_names)

        return data

    def get_dataloader(self, split, batchSize):
        data = self.process_split(split, batchSize)

        return DataLoader(data, batch_size=batchSize, shuffle=(split=="train"), collate_fn=self.collator) 

if __name__ == "__main__":
    dataSet = BioLaySummData()
    dataSet.setup_default()

    trainData = dataSet.get_dataloader("train", 512)
    valData = dataSet.get_dataloader("validation", 512)
    testData = dataSet.get_dataloader("test", 512)

