# Next Word Prediction Using Deep Learning

## About the Project

This project focuses on building a next-word prediction system using Natural Language Processing (NLP) and deep learning.

The goal is to predict the next word based on a given sequence of previous words.

## Dataset

We use the **WikiText-2 dataset** from Hugging Face.

- Vocabulary size: 10,000
- Context window: 5 words
- Tokenization: Keras Tokenizer
- Padding: Pre-padding

## Dataset Preparation

The following steps have been completed:

1. Dataset collection
2. Text cleaning and normalization
3. Sentence splitting
4. Tokenization
5. Input–target sequence generation
6. Padding
7. Training, validation, and testing data preparation
8. Dataset verification

## Dataset Statistics

| Dataset | Samples |
|---|---:|
| Training | 1,668,886 |
| Validation | 175,071 |
| Testing | 196,425 |

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- NLTK
- Hugging Face Datasets
- Git and GitHub

## Team Responsibilities

- **Member 1:** Dataset collection and preprocessing — Completed
- **Member 2:** Model development and training
- **Member 3:** Model evaluation and next-word prediction

## Project Status

Data preparation and verification are complete. Model development and evaluation are the next stages.

## Documentation

For detailed preprocessing instructions, see [Data Preparation Documentation](docs/data_preparation.md).
