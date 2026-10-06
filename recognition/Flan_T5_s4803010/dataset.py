import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

class BioLaySummData():
    def __init__(self):
        self.dataPath = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
        self.data = None

        self.maxInputTokens = 512
        self.maxLabelTokens = 512

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

if __name__ == "__main__":
    dataSet = BioLaySummData()
    dataSet.setup_default()

    trainData = dataSet.process_split("train", 512)
    valData = dataSet.process_split("validation", 512)
    testData = dataSet.process_split("test", 512)

    trainLoader = DataLoader(trainData, batch_size=16, shuffle=True, collate_fn=dataSet.get_collator())
    foo = next(iter(trainLoader))