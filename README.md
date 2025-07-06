# NLPFinanceCode (BERT-Based Sentiment Classifier)

This project builds a sentiment classification model using BERT (Bidirectional Encoder Representations from Transformers) to analyze financial news headlines. It classifies each headline as positive, neutral, or negative, offering insights into how sentiment might relate to stock market movements.

## Overview

The notebook walks through the complete pipeline of fine-tuning a pre-trained BERT model using PyTorch and Hugging Face's transformers library.

## Steps

### 1. Data Preparation
- Loads pre-labeled datasets of financial headlines for training, validation, and testing.
- Each headline is associated with a sentiment label: 0 (negative), 1 (neutral), or 2 (positive).

### 2. Text Preprocessing
- Headlines are tokenized using BERT's tokenizer.
- Special tokens [CLS] and [SEP] are added.
- Tokenized sentences are padded to a uniform length.
- Attention masks are created to differentiate real tokens from padding.

### 3. Model Configuration
- Utilizes BertForSequenceClassification with 3 output classes.
- Fine-tuned using the AdamW optimizer and linear learning rate scheduler.
- Trained using GPU acceleration when available.

### 4. Training & Validation
- The model is trained for 4 epochs with batch size 32.
- Tracks training and validation loss per epoch.
- Evaluation metrics include accuracy and loss.

### 5. Testing
- The test dataset is preprocessed similarly and fed into the trained model.
- Final evaluation includes test accuracy and a confusion matrix.

## Results

- Achieves strong performance in multi-class sentiment classification.
- Training/validation losses are plotted for analysis.
- Confusion matrix visualizes how well the model distinguishes between sentiment classes.

## Technologies Used

- Python, PyTorch
- Hugging Face transformers library
- BERT-base-uncased model
- scikit-learn, pandas, matplotlib, seaborn

## How to Run

1. Clone the repository and install dependencies:
   ```bash
   pip install transformers torch scikit-learn pandas matplotlib seaborn
