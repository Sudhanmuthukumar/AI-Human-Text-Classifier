# AI Human Text Classifier

## Project Overview

AI Human Text Classifier is a machine learning and NLP-based web application developed to distinguish between AI-generated and human-written text.

The project combines text preprocessing, feature engineering, machine learning models, and a Flask web application to provide predictions along with additional text analysis such as readability, vocabulary diversity, and sentiment analysis.

This project was developed as a personal machine learning project to explore Natural Language Processing (NLP), ensemble learning, and web deployment.

---

## Features

* Classifies text as AI-generated or Human-written
* Displays confidence scores for predictions
* Performs sentiment analysis
* Calculates readability scores
* Measures vocabulary diversity
* Computes average sentence length
* Interactive Flask-based web interface

---

## Machine Learning Pipeline

1. Text preprocessing using NLTK
2. Stopword removal and lemmatization
3. TF-IDF vectorization
4. Dimensionality reduction using Truncated SVD
5. Linguistic feature extraction
6. Classification using:

   * XGBoost
   * Logistic Regression
7. Ensemble prediction for final output

---

## Model Performance

The model was evaluated on a separate test dataset and achieved:

* Accuracy: **98.68%**
* F1 Score: **0.9822**

### Confusion Matrix

| Actual | Predicted Human | Predicted AI |
| ------ | --------------- | ------------ |
| Human  | 60,768          | 392          |
| AI     | 892             | 35,395       |

A confusion matrix image is included in this repository.

---

## Dataset

This project uses the AI vs Human Text Dataset available on Kaggle:

https://www.kaggle.com/datasets/akellahima/ai-human

The dataset is not included in this repository because of GitHub file size limitations.

After downloading the dataset, place it inside:

```text
data/AI_Human.csv
```

---

## Technologies Used

* Python
* Flask
* Scikit-learn
* XGBoost
* NLTK
* Pandas
* NumPy
* HTML
* CSS
* JavaScript

---

## Project Structure

```text
AI-Human-Text-Classifier
│
├── app.py
├── train_model.py
├── README.md
├── requirements.txt
├── confusion_matrix.png
│
├── models
│   ├── tfidf.pkl
│   ├── svd.pkl
│   ├── log_reg.pkl
│   └── xgb_booster.json
│
├── static
├── templates
└── data
```

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the Flask application:

```bash
python app.py
```

---

## Future Improvements

* Deep learning-based text classification
* Support for additional languages
* Improved explainability and feature visualization
* Cloud deployment
* Larger and more diverse datasets

---

## Author

Sudhan M

B.E. Computer Science and Engineering

Interested in Machine Learning, NLP, AI Systems, and Software Development.
