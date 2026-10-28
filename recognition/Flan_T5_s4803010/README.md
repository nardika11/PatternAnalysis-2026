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
### 7.1 User Need, Scope and Acceptance Criteria
Medical reports can be confusing for patients to understand which can cause anxiety. The purpose of this project is to assist with the translation of medical reports into layman language to increase readability for the general public who may not have medical knowledge. Within the project, experimentation will be conducted to investigate whether fine-tuning a model preserves more medical accuracy compared to a zero-shot baseline model. 

Acceptance Criteria:
- FLAN-T5 models are successfully trained using both LoRA and full fine-tuning 
- Fine tuning improves Rouge evaluation metric compared to a zero-shot baseline
- A manual evaluation is conducted on sample cases to identify errors such as hallucinations, missing information and layman readability
- An evaluation on the trade-offs between Rouge metrics and clinical accuracy/readability has been conducted
- Training is able to be completed using the rangpur A100 cluster in under 5 hours each, with training metrics such as runtime, peak VRAM and parameter counts being documented

### 7.2 Model Choice and Course Concepts
Google's FLAN-T5 encoder-decoder model will be the foundation of each model variation that is experimented on for the project, which is a hard difficulty. FLAN-T5 is an instruction based sequence to sequence model that has been trained for over 1000 NLP tasks which also includes summarisation. Additionally, the FLAN-T5 architecture is well suited for this projects task, since the encoder component allows the model to understand and create representations for the medical report, while the decoder component can then decode these representations in order to generate layman summaries of the original text. 

The original FLAN-T5 will be used as a zero-shot untuned baseline model, to compare against the two fine-tuned variations. The second variation will use LoRA fine-tuning, which tunes a subset of the models parameters in order to reduce the computational resources required. The last variation uses full fine-tuning in order to tune all parameters based on the training set. Using the two versions of fine-tuning allows for the investigation between the required computational requirements and the summarisation quality. 

### 7.3 Preliminary Feasibility Evidence
The BioLaySumm dataset provided already has predefined train, validation and test splits with the following respective proportions 88%, 5.8% and 6.2%, so in order to prevent data leakage these splits will be preserved throughout the project. Each datapoint has four features, source, images_path, radiology_report and layman_report. For the project only the last two features are relevant and will be used, radiology_report as the input and layman_report as the target.

The tokenised inputs and labels need to be truncated to a max token length in order to assist with limiting the computational resources (specifically GPU memory and processing time) needed to train and run the models. Initial experimentation has been completed which found that the 99th percentile for token size for the inputs size was 300.47 and 296 for the labels. It was also found that for the inputs the max token length was 1742 with a median of 47, and for the labels these respective values were 1176 and 41. This suggests that a max token length of 512 is suitable as it ensures that over 99% of the inputs and labels never get truncated whilst only truncating some rare cases over 512 tokens. This ensures loss of data is minimised while still getting the benefit of computational resource reduction. 

Another key finding within the above experimentation is that the median token lengths were very low (47 and 41) compared to the truncation limit that was just formed. Because of this static padding to 512 creates a large amount of useless padding which increases computational costs. To solve this dynamic padding can be used which will only pad inputs/labels to the max token sequence length within the current batch. Since most sequences are roughly 50 tokens in length, most batches shouldn't use large amounts of useless padding.  

### 7.4 Risks, Budget and Fallback
Due to the large computational requirements (plus possible Rangpur wait times) to perform full fine tuning on the FLAN-T5 model, there is a risks that the full fine tuning isn't able to be completed in time. A possible fallback would be to use the FLAN-T5 small version rather than the base model, which only has 80 million parameters rather than 250 million. This should reduce the computational requirements needed to perform full fine tuning and could increase the feasibility to compute on my local machine rather than queuing for Rangpur. 

Another risk is that the fine tuned model perform worse on the ROUGE evaluation metrics compared to the baseline model. This would make it hard to answer the project question of whether fine tuned models improve medical context summarisation or just plain english summarisation. 

After the baseline model completion for the feasibility test, the next step is to implement the module and setup required for LoRA fine tuning. After this the full fine tuning will be trained and then all three versions will be evaluated against each other. 

## 8. AI Usage Disclosure

## 9. References