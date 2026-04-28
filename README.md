# 🧠 Emotion Detector — NLP Text Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLTK-3.8+-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  A machine learning web application that detects human emotions from text using Natural Language Processing.
  Built with TF-IDF vectorization and Logistic Regression, wrapped in an interactive Streamlit UI.
</p>

🔗 **Live Demo:** [emotion-detector-pranjal-pandey.streamlit.app](https://emotion-detector-pranjal-pandey.streamlit.app/)
---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Emotions Detected](#-emotions-detected)
- [Project Structure](#-project-structure)
- [NLP Pipeline](#-nlp-pipeline)
- [Model Comparison](#-model-comparison)
- [Tech Stack](#-tech-stack)
- [Installation & Usage](#-installation--usage)
- [Dataset](#-dataset)
- [Results](#-results)
- [Author](#-author)

---

## 🔍 Overview

This project builds an end-to-end NLP pipeline that classifies text into one of **6 human emotions**. The pipeline includes full text preprocessing, feature extraction using TF-IDF and Bag of Words, and classification using multiple ML models. The best model is served through a **Streamlit web app** with a clean, interactive UI.

---

## 🎬 Demo

> Run locally with:
```bash
python -m streamlit run emotion_app.py
```

**Features of the UI:**
- Upload your own training data
- Type any text and get instant emotion prediction
- View confidence scores for all 6 emotions as a bar chart
- See cleaned text after the full preprocessing pipeline
- View training data emotion distribution chart

---

## 🎭 Emotions Detected

| Emotion | Label | Emoji |
|---------|-------|-------|
| Sadness | 0 | 😢 |
| Anger | 1 | 😠 |
| Love | 2 | ❤️ |
| Surprise | 3 | 😲 |
| Fear | 4 | 😨 |
| Joy | 5 | 😄 |

---

## 📁 Project Structure

```
emotion-detector-nlp/
│
├── emotion_app.py          # Streamlit web application
├── NLP-emotions.ipynb      # Full model training notebook
├── train.txt               # Training dataset (text;emotion format)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🔄 NLP Pipeline

Every input text goes through the following preprocessing steps before prediction:

```
Raw Text
   │
   ▼
1. Lowercase conversion         → "I AM HAPPY" → "i am happy"
   │
   ▼
2. Remove punctuation           → "hello!!!" → "hello"
   │
   ▼
3. Remove numbers               → "room 404" → "room "
   │
   ▼
4. Remove emojis                → keeps only ASCII characters
   │
   ▼
5. Remove URLs                  → strips http/https/www links
   │
   ▼
6. Remove stopwords (NLTK)      → removes "is", "the", "and" etc. (198 words)
   │
   ▼
Cleaned Text → TF-IDF Vectorizer → Logistic Regression → Predicted Emotion
```

---

## 📊 Model Comparison

Three models were trained and evaluated. Here are the results:

| Model | Vectorizer | Test Accuracy | Notes |
|-------|-----------|---------------|-------|
| Multinomial Naive Bayes | Bag of Words (CountVectorizer) | ~77% | Fast, but lower accuracy |
| Multinomial Naive Bayes | TF-IDF | ~74% | Worse than BoW for this dataset |
| **Logistic Regression** | **TF-IDF** | **~86.2%** | ✅ Best model — selected for deployment |

### Why Logistic Regression won?
- Handles high-dimensional sparse TF-IDF features well
- Outputs calibrated probabilities for all classes
- `max_iter=1000` allowed full convergence of the optimizer
- Significantly outperformed Naive Bayes on this 6-class problem

### Why TF-IDF over Bag of Words?
- TF-IDF penalizes very common words that appear across all documents
- Gives higher weight to words that are more unique to specific emotions
- Reduces noise from frequently occurring but uninformative words

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Core programming language |
| **Pandas** | Data loading and manipulation |
| **NLTK** | Stopwords removal and tokenization |
| **Scikit-learn** | TF-IDF, CountVectorizer, Logistic Regression, Naive Bayes |
| **Matplotlib** | Probability bar charts and distribution plots |
| **Streamlit** | Interactive web UI |
| **NumPy** | Numerical operations |

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/venom312004/emotion-detector-nlp.git
cd emotion-detector-nlp
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python -m streamlit run emotion_app.py
```

### 4. Use the app
- Upload your `train.txt` file via the sidebar
- Wait a few seconds for the model to train
- Type any text in the input box
- Click **Predict Emotion** and see the result!

---

## 📂 Dataset

The dataset is a plain text file (`train.txt`) with semicolon-separated values:

```
i feel really happy today;joy
i miss you so much;sadness
this makes me so angry;anger
i love spending time with you;love
i had no idea this would happen;surprise
i am terrified of the dark;fear
```

- **Format:** `text;emotion`
- **Classes:** 6 (sadness, anger, love, surprise, fear, joy)
- **Split:** 80% training / 20% testing
- **Random state:** 42 (reproducible results)

---

## 📈 Results

```
Best Model   : Logistic Regression + TF-IDF
Test Accuracy: ~86.2%
Classes      : 6
Max Features : 20,000 (TF-IDF vocabulary)
Max Iter     : 1000
Solver       : lbfgs
```

The model performs particularly well on **joy**, **sadness**, and **anger** — emotions with strong distinctive vocabulary. **Surprise** and **fear** sometimes overlap due to similar language patterns.

---

## 👤 Author

**Pranjal Pandey**
- 🎓 B.Tech in Artificial Intelligence
- 💼 Email: pranjalpandey0301@gmail.com
- 🐙 GitHub: [@venom312004](https://github.com/venom312004)

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute.

---

<p align="center">Made with ❤️ using Python & Streamlit</p>
