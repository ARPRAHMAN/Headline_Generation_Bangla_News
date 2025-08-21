# 📰 Bangla News Headline Generation using Seq2Seq and Transformer Models  

![Python](https://img.shields.io/badge/Python-3.11-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange)  
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)  
![License](https://img.shields.io/badge/License-MIT-green)  

---

## 📖 Overview  

This project implements **automatic headline generation for Bangla news articles** using **deep learning models**.  
We compare **Seq2Seq architectures (LSTM, BiLSTM)** with a **custom Transformer** to generate concise, meaningful Bangla headlines from longer news content.  

The pipeline includes:  
- **Text cleaning & normalization** (BUET Normalizer)  
- **Tokenizer & sequence padding**  
- **Model training** (LSTM, BiLSTM, Transformer)  
- **Evaluation** using ROUGE scores  
- **Visualization** of training & validation loss  

---

## 📂 Project Structure  
Bangla-Headline-Generation/
├── data/ # Dataset (Bangla news CSV)
├── models/ # Trained model files (optional)
├── notebooks/ # Jupyter/Colab notebooks
├── results/ # Logs, graphs, evaluation outputs
└── README.md # Project documentation


---

## 🧰 Requirements  

Install dependencies before running:  

```bash
pip install git+https://github.com/csebuetnlp/normalizer
pip install nltk pandas numpy matplotlib scikit-learn tensorflow transformers sentencepiece rouge_score
```

## 🚀 Workflow
1. Data Preprocessing

Load Bangla news dataset (content → article, title → headline)

Clean & normalize using BUET Normalizer

Split into train (80%), validation (10%), and test (10%)

2. Tokenization & Padding

Tokenize both content and title

Define maximum lengths:

Content: 150 tokens

Title: 20 tokens

3. Model Architectures

LSTM Seq2Seq: Encoder-Decoder with hidden states

BiLSTM Seq2Seq: Bidirectional encoder for richer context

Custom Transformer: Multi-head attention + feed-forward network

4. Training

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Epochs:

LSTM: 20

BiLSTM: 20

Transformer: 10

5. Evaluation

Metric: ROUGE-1, ROUGE-2, ROUGE-L

Compare performance of LSTM, BiLSTM, Transformer

Plot training & validation loss curves


## 🔍 Key Features

✅ End-to-End Bangla news headline generation

✅ Implements Seq2Seq & Transformer architectures

✅ Integrated with BUET Normalizer for Bangla text

✅ Evaluation with ROUGE metrics

✅ Visual training curves for model comparison

## ⚠️ Limitations

Current models generate syntactically valid but semantically weak headlines

ROUGE scores are poor due to limited vocabulary handling

Future work:

Use Bangla-BERT / mBERT embeddings

Apply beam search decoding

Train on larger datasets
