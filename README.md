# Fake Review Detection using NLP & Deep Learning 

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-red)
![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20SpaCy-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview

Online reviews significantly influence customer purchasing decisions. However, the increasing presence of fake and spam reviews can mislead consumers and reduce trust in e-commerce platforms.

This project leverages **Natural Language Processing (NLP)** and **Machine Learning / Deep Learning** techniques to automatically classify reviews as **Real (Genuine)** or **Fake (Spam)**.

The system processes textual reviews, extracts meaningful features, trains classification models, and predicts the authenticity of new reviews.

---

## 🎯 Objectives

* Detect fake and genuine reviews automatically
* Apply NLP techniques for text preprocessing
* Extract features using TF-IDF vectorization
* Train Machine Learning and Deep Learning models
* Evaluate performance using standard metrics
* Predict authenticity of unseen reviews

---

## 🏗️ System Architecture

```text
                 ┌─────────────┐
                 │   Dataset   │
                 └──────┬──────┘
                        │
                        ▼
             ┌──────────────────┐
             │ Text Preprocessing│
             └──────┬───────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │ TF-IDF Vectorization│
          └──────┬──────────────┘
                 │
                 ▼
          ┌─────────────────────┐
          │ Model Training      │
          └──────┬──────────────┘
                 │
                 ▼
          ┌─────────────────────┐
          │ Model Evaluation    │
          └──────┬──────────────┘
                 │
                 ▼
          ┌─────────────────────┐
          │ Review Prediction   │
          └─────────────────────┘
```

---

## 📂 Dataset

### Source

Amazon Reviews Dataset

### Features

| Feature           | Description                  |
| ----------------- | ---------------------------- |
| Review Text       | User review content          |
| Rating            | Product rating (1–5 stars)   |
| Reviewer ID       | Reviewer identifier          |
| Product ID        | Product identifier           |
| Verified Purchase | Purchase verification status |

### Target Labels

| Label | Meaning        |
| ----- | -------------- |
| 0     | Genuine Review |
| 1     | Fake Review    |

---

## 🧹 Text Preprocessing

Before training, reviews undergo multiple cleaning steps:

### Preprocessing Pipeline

✅ Convert text to lowercase

✅ Remove stopwords

✅ Tokenization

✅ Stemming / Lemmatization

✅ Remove punctuation

✅ Remove special characters

### Example

**Input**

```text
"This product is AMAZING!!!"
```

**Output**

```text
product amazing
```

---

## 🔍 Feature Extraction

### TF-IDF Vectorization

TF-IDF (Term Frequency – Inverse Document Frequency) converts textual data into numerical vectors that machine learning models can understand.

### Formula

```text
TF-IDF = TF × IDF
```

### Advantages

* Highlights important words
* Reduces influence of common words
* Improves model performance

---

## 🤖 Models Used

### Machine Learning Models

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)

### Deep Learning Models

* Recurrent Neural Network (RNN)
* Long Short-Term Memory (LSTM)
* BERT Transformer Model

---

## 📊 Model Training

### Training Configuration

```text
Training Data : 80%
Testing Data  : 20%
```

### Workflow

```text
Review Text
     │
     ▼
Preprocessing
     │
     ▼
TF-IDF Features
     │
     ▼
Model Training
     │
     ▼
Classification
```

---

## 📈 Evaluation Metrics

The model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

### Confusion Matrix Example

| Actual | Predicted | Result      |
| ------ | --------- | ----------- |
| Fake   | Fake      | ✅ Correct   |
| Real   | Fake      | ❌ Incorrect |
| Real   | Real      | ✅ Correct   |
| Fake   | Real      | ❌ Incorrect |

---

## 🔮 Prediction Examples

### Example 1

**Input**

```text
Excellent product!!! Must buy!!!
```

**Prediction**

```text
Fake Review
Confidence: 87%
```

---

### Example 2

**Input**

```text
I used this product for 2 weeks. Battery life is good but camera quality is average.
```

**Prediction**

```text
Genuine Review
Confidence: 91%
```

---

## 🧠 Detection Logic

### Indicators of Fake Reviews

* Excessive exclamation marks
* Generic statements
* Repeated words
* Lack of personal experience
* Overly promotional language

### Indicators of Genuine Reviews

* Detailed explanations
* Balanced feedback
* Personal experiences
* Natural writing style
* Mention of pros and cons

---

## 🛠️ Technologies Used

| Category        | Technologies        |
| --------------- | ------------------- |
| Programming     | Python              |
| NLP             | NLTK, SpaCy         |
| ML              | Scikit-Learn        |
| Deep Learning   | TensorFlow, Keras   |
| Data Processing | Pandas, NumPy       |
| Visualization   | Matplotlib, Seaborn |

---

## 📁 Project Structure

```text
fake-review-detection/
│
├── data/
│   ├── reviews.csv
│
├── models/
│   ├── trained_model.pkl
│
├── notebooks/
│   ├── experimentation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── predict.py
│
├── main.py
├── requirements.txt
├── README.md
│
└── outputs/
    ├── results.png
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/fake-review-detection.git
```

### 2️⃣ Navigate to Project Directory

```bash
cd fake-review-detection
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Application

```bash
python main.py
```

---

## 📌 Sample Workflow

```text
User Review
      │
      ▼
Text Cleaning
      │
      ▼
TF-IDF Conversion
      │
      ▼
Trained Model
      │
      ▼
Fake / Genuine Prediction
```

---

## 🚀 Future Enhancements

* Fine-tune BERT and RoBERTa models
* Real-time review monitoring
* React/MERN web interface
* Cloud deployment (AWS/Azure)
* Browser extension integration
* Multilingual fake review detection
* Explainable AI (XAI) support

---

## ⚠️ Limitations

* Performance depends on dataset quality
* Advanced fake reviews can mimic genuine reviews
* Contextual sarcasm may be misclassified
* Requires periodic retraining for evolving patterns

---

## 🎓 Learning Outcomes

Through this project, you will gain experience in:

* Natural Language Processing (NLP)
* Text Classification
* Feature Engineering
* Machine Learning Pipelines
* Deep Learning for NLP
* Model Evaluation Techniques

---

## ✅ Conclusion

The **Fake Review Detection System** demonstrates how NLP and AI techniques can be utilized to identify deceptive reviews and improve trust in online platforms. By combining text preprocessing, TF-IDF feature extraction, and classification models, the system provides an effective solution for distinguishing between genuine and fake reviews.

---

### 👩‍💻 Developed By

**Komal Khatod**
Computer Science Engineering Student | AI/ML Enthusiast | MERN Developer

⭐ If you found this project useful, don't forget to star the repository!
