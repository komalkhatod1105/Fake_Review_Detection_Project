# Fake Review Detection using NLP, Machine Learning & Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![NLP](https://img.shields.io/badge/NLP-TextProcessing-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

Fake reviews have become a major challenge for e-commerce platforms, influencing customer purchasing decisions and reducing trust in online marketplaces.

This project leverages **Natural Language Processing (NLP)**, **Machine Learning**, and **Deep Learning** techniques to automatically identify whether a review is genuine or fraudulent.

The system analyzes review text, extracts meaningful linguistic patterns, and classifies reviews as **Real** or **Fake** with high accuracy.

---

## Key Features

- Automated Fake Review Detection
- NLP-Based Text Processing
- TF-IDF Feature Extraction
- Machine Learning Models
- Deep Learning Integration
- Real-Time Review Classification
- Performance Evaluation Metrics

---

## Problem Statement

Online platforms receive thousands of reviews every day. Many of these reviews are artificially generated to:

- Increase product ratings
- Manipulate customer opinions
- Damage competitors' reputation
- Influence purchasing decisions

This project aims to build an intelligent system capable of distinguishing genuine customer feedback from deceptive reviews.

---

## Project Workflow

```text
Dataset
   │
   ▼
Text Preprocessing
   │
   ▼
Feature Extraction (TF-IDF)
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Prediction
```

---

## Dataset

### Source

Amazon Product Reviews Dataset

### Features

| Feature | Description |
|----------|-------------|
| Review Text | Customer review content |
| Rating | Product rating (1–5 stars) |
| Reviewer ID | Unique reviewer identifier |
| Product ID | Product identifier |
| Verified Purchase | Purchase authenticity flag |

### Labels

| Label | Meaning |
|---------|---------|
| 0 | Genuine Review |
| 1 | Fake Review |

---

## Text Preprocessing

Raw review text undergoes several preprocessing steps before model training.

### Steps

- Convert text to lowercase
- Remove stopwords
- Tokenization
- Lemmatization
- Remove punctuation
- Remove special characters
- Text normalization

### Example

Input:

```text
"This product is AMAZING!!!"
```

Output:

```text
product amazing
```

---

## Feature Engineering

### TF-IDF Vectorization

TF-IDF (Term Frequency–Inverse Document Frequency) converts textual reviews into numerical vectors suitable for machine learning algorithms.

### Benefits

- Highlights important words
- Reduces impact of common words
- Improves classification performance
- Efficient for large datasets

---

## Machine Learning Models

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest

---

## Deep Learning Models

- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Bidirectional LSTM
- BERT Transformer

---

## Model Training

### Training Configuration

```text
Training Data : 80%
Testing Data  : 20%
```

### Input

```text
TF-IDF Feature Vectors
```

### Output

```text
Real Review
or
Fake Review
```

---

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Sample Evaluation

| Actual | Predicted | Result |
|----------|------------|---------|
| Fake | Fake | Correct |
| Real | Fake | Incorrect |
| Real | Real | Correct |

---

## Prediction Examples

### Example 1

Input:

```text
Excellent product!!! Must buy!!!
```

Prediction:

```text
Fake Review
Confidence Score: 87%
```

### Example 2

Input:

```text
I used this product for 2 weeks. Battery backup is excellent, but the camera quality could be improved.
```

Prediction:

```text
Genuine Review
Confidence Score: 91%
```

---

## Fake Review Indicators

The model identifies common patterns in fake reviews:

- Excessive promotional language
- Repetitive wording
- Too many exclamation marks
- Generic descriptions
- Lack of personal experience

Example:

```text
Best product ever!!!
Amazing!!!
Highly recommended!!!
```

---

## Genuine Review Indicators

The model identifies characteristics of authentic reviews:

- Detailed explanations
- Personal experiences
- Balanced opinions
- Natural writing style
- Mention of both pros and cons

Example:

```text
The battery life is impressive, but charging speed is slightly slower than expected.
```

---

## Technology Stack

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### NLP Libraries

- NLTK
- SpaCy

### Machine Learning

- Scikit-Learn

### Deep Learning

- TensorFlow
- Keras
- Transformers

### Visualization

- Matplotlib
- Seaborn

---

## Project Structure

```text
Fake-Review-Detection/
│
├── dataset/
│   └── reviews.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── models/
│   └── trained_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Project Screenshot

![Project Structure](https://github.com/user-attachments/assets/183209a7-06ba-4aa5-863e-e25ba4a5ab1c)

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Fake_Review_Detection_Project.git
```

### Move to Project Directory

```bash
cd Fake_Review_Detection_Project
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

---

## Future Enhancements

- Fine-Tuned BERT Models
- Real-Time Review Monitoring
- Browser Extension Integration
- MERN Stack Dashboard
- Multilingual Review Analysis
- Cloud Deployment
- Explainable AI (XAI)

---

## Challenges & Limitations

- Dataset quality affects accuracy
- Fake reviews are becoming increasingly realistic
- Domain-specific reviews may require retraining
- Language variations can impact predictions

---

## Results

This project demonstrates how NLP, Machine Learning, and Deep Learning can effectively identify deceptive online reviews and improve trust in digital marketplaces.

---

## Author

### Komal Khatod

B.Tech Computer Science Engineering  
Mody University of Science and Technology

GitHub: https://github.com/komalkhatod1105

---



⭐ If you found this project useful, consider giving it a star on GitHub.