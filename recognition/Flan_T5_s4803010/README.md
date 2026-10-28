# Radiology Report Summarisation using FLAN-T5: Investigating the Effects of Full Fine-Tuning and LoRA 

## 1. Overview
### 1.1 Problem Description
Radiology reports quite often contain complex wording and specialised medical terms. This can cause patients to not fully understand their results, potentially leading to confusion and higher anxiety. Using simple translation methods to summaries the reports can reduce this issue. However there is a high risk of translation/generation errors such as stating incorrect information, missing key details and distorting clinical certainty.

### 1.2 Project Objective
This project aims to investigate how Google's FLAN-T5 model, through fine-tuning, can increase both the layman readability of reports whilst still maintaining clinical accuracy on the BioLaySumm 2025 dataset. Three model configurations were compared within the project including an untuned baseline, full fine-tuning and Low-Rank Adaptation (LoRA) fine-tuning. The models were evaluated lexicographically using ROUGE scores, computationally using metrics such as peak VRAM and training time, and clinically through a manually safety audit. The main objective of the project was to explore whether improvements in lexical similarity (through ROUGE) correlates to better clinical factual accuracy, or whether fine-tuned models still contain potential harmful errors. 

## 2. Algorithm and Architecture 
### 2.1 FLAN-T5 Architecture
Google's FLAN-T5-base model is used as the foundational model of the project. It is an instruction tuned Large Language model which is suited for multiple natural language problems such as translation, summarisation and question answering. It uses an encoder-decoder transformer architecture where the encoder processes the input text sequence and the decoder generates the corresponding output sequence. 

Although FLAN-T5 is well suited for general summarisation, it is not optimised specifically for clinical radiology reports, which is where fine-tuning comes in. Fine-tuning the model using the BioLaySumm dataset allows the model to better learn medical terminology in order to produce summaries that can be better understood by patients, whilst still preserving key clinical information.  

### 2.2 Full Fine-Tuning
Full fine-tuning involves updating all of the models parameters during training in order to get better performance specifically using the BioLaySumm training dataset. During the forward pass, the inputs are processed through the model, while backpropagation calculates the gradients in order to update entire model's weight matrix. However this approach can be more computationally expensive, requiring higher GPU memory usage in order to calculate/store gradients and update parameters.

### 2.3 Low-Rank Adaptation (LoRA)
Low-Rank Adaptation aims to make fine-tuning more efficient by reducing the number of parameters that are updated during training. This is done by freezing the original model's weight matrix and introduces two smaller matrices within selected layers. These matrices are updated during training and then are merged with the original weight matrix after training to produce the final model. Similarly to full fine-tuning, a forward pass is performed on the original weight matrix (plus the smaller matrices for LoRA), however backpropagration is only performed on the smaller matrices. 

By training on just the smaller two matrices, LoRA significantly reduces the amount of trainable parameters, therefore reducing the computational requirements. However by limiting the parameter updated to a smaller subset, the model's ability to learn clinical summarisation may be limited compared to full fine-tuning.

## 3. Dataset and Preprocessing
### 3.1 BioLaySummm Dataset
### 3.2 Data Preprocessing
### 3.3 Data Splits

## 4. Implementation and Training
### 4.1 File Structure
### 4.2 Training Configurations
### 4.3 Evaluation Process
### 4.4 Hardware Configurations
### 4.5 Installation and Usage

## 5. Experimental Results
### 5.1 Training and Validation Comparison
### 5.2 Summarisation Comparison
### 5.3 Computational Comparison

## 6. Open Research Dilemma 
### 6.1 Clinical Safety Autopsy 
### 6.2 ROUGE vs Clinical Factual Accuracy
### 6.3 Engineering Recommendations

## 7. Feasibility Review

## 8. AI Usage Disclosure

## 9. References