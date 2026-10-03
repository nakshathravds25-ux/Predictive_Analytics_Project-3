# Email Spam Classification — NLP, ML & LSTM

An end-to-end NLP project for classifying email text as **spam** or **ham (legitimate)** using classical machine-learning and deep-learning approaches.

## What this project demonstrates

- Email-text preprocessing
- TF-IDF feature engineering
- Naive Bayes and Support Vector Machine baselines
- Bidirectional LSTM sequence modelling
- Model comparison and evaluation
- Saved model artefacts for inference
- Streamlit deployment

## Dataset

The project uses public email corpora including the **SpamAssassin Public Corpus** and **Enron email data**.

Raw text is cleaned and converted into model-ready representations while preserving the binary spam/ham target.

## Workflow

```text
Raw email text
      ↓
Cleaning and preprocessing
      ↓
 ┌────────────────────┬─────────────────────┐
 │ TF-IDF             │ Token sequences     │
 │                    │                     │
 │ Naive Bayes / SVM  │ Bidirectional LSTM  │
 └────────────────────┴─────────────────────┘
      ↓
Held-out evaluation
      ↓
Streamlit inference app
```

## Models

### Naive Bayes
A lightweight probabilistic baseline trained on TF-IDF features.

### Support Vector Machine
A linear text classifier trained on sparse TF-IDF features.

### Bidirectional LSTM
A neural sequence model using tokenized and padded email text.

## Evaluation

The notebooks evaluate the models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For portfolio and resume use, model metrics should be quoted only from a reproducible held-out evaluation run. The repository therefore focuses on the pipeline and implementation rather than treating previously reported coursework numbers as universal benchmark results.

## Repository Structure

```text
Predictive_Analytics_Project-3/
├── app.py
├── feature_extraction.py
├── Preprocess.ipynb
├── model_training_py.ipynb
├── Email_spam_classification_(1) (1).ipynb
├── trained_models/
├── screenshots/
├── requirements.txt
└── README.md
```

## Run the App

```bash
git clone https://github.com/nakshathravds25-ux/Predictive_Analytics_Project-3.git
cd Predictive_Analytics_Project-3
pip install -r requirements.txt
streamlit run app.py
```

## Live Demo

https://email-spam-detection-i3plduappn4kgf8p6rwfzzn.streamlit.app/

## Team Contribution

This was a team course project.

- **Nakshathra V:** feature engineering and model training
- Krishnanjana J J: data collection, preprocessing and EDA
- Harikrishnan S M: deployment and documentation

## Tech Stack

Python · Pandas · NumPy · scikit-learn · TensorFlow/Keras · NLTK · Streamlit

## Limitations & Future Work

Potential improvements include stronger duplicate/leakage checks across source corpora, threshold tuning, calibration, transformer baselines, explainability, and testing on temporally newer spam distributions.
