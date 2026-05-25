# 🎓 Student Risk Predictor

> An AI-powered student academic risk prediction system that identifies at-risk students early and provides personalized intervention recommendations — built with Streamlit and trained on three ML models.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [How It Works](#how-it-works)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Risk Levels](#risk-levels)
- [Personalized Recommendations](#personalized-recommendations)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)

---

## Overview

Student Risk Predictor uses machine learning to predict whether a student is academically at risk based on five key performance indicators. It supports early intervention by flagging struggling students before it's too late — with an interactive Streamlit dashboard, explainable AI feature importance charts, and a built-in AI chatbot.

---

## Project Structure

```
student-risk-predictor/
├── app.py                  # Streamlit dashboard — 3 tabs
├── model.py                # Training script — trains & saves all 3 models
├── model_comparison.csv    # Accuracy results for all 3 models
├── stddataset.xlsx         # Student dataset
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore
```

> **Note:** `.pkl` model files are not included. Run `model.py` first to generate them.

---

## Features

- **3 ML models** — Logistic Regression, Random Forest, XGBoost
- **Dashboard overview** — total students, at-risk count, risk distribution chart
- **Individual student prediction** — enter 5 features, get instant risk verdict
- **3-tier risk levels** — High, Medium, Low with color coding
- **Personalized recommendations** — tailored advice based on weak indicators
- **Explainable AI** — feature importance charts for Random Forest and XGBoost
- **AI Chatbot** — answers questions about risk factors and improvement strategies
- **Download results** — export all predictions as CSV

---

## How It Works

```
stddataset.xlsx
      │
      ▼
model.py  ──►  trains 3 models  ──►  saves .pkl files
      │
      ▼
app.py (Streamlit)
  ├── Tab 1: Dashboard Overview    → risk summary + distribution chart
  ├── Tab 2: Individual Prediction → input features → risk level + recommendations
  └── Tab 3: Explainable AI        → feature importance + AI chatbot
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/student-risk-predictor.git
cd student-risk-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models

```bash
python model.py
```

This generates:
- `logistic_regression_model.pkl`
- `random_forest_model.pkl`
- `xgboost_model.pkl`
- `model_features.pkl`
- `model_comparison.csv`

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Risk Levels

| Level | Probability | Meaning |
|-------|------------|---------|
| 🔴 HIGH RISK | ≥ 75% | Immediate intervention required |
| 🟠 MEDIUM RISK | 45% – 75% | Student needs close monitoring |
| 🟢 LOW RISK | < 45% | Performance indicators are satisfactory |

---

## Personalized Recommendations

Based on the student's input, the system automatically suggests targeted actions:

| Weak Indicator | Recommendation |
|---------------|----------------|
| Attendance < 75% | Improve attendance with weekly monitoring |
| Assignment Completion < 60% | Provide extra practice assignments |
| Study Hours < 10 | Introduce time-management strategies |
| Exam Score < 60 | Arrange remedial classes |

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 82.0% | 77.1% | 81.9% | 79.5% |
| Random Forest | 100% | 100% | 100% | 100% |
| XGBoost | 100% | 100% | 100% | 100% |

---

## Input Features

| Feature | Description |
|---------|-------------|
| Study Hours | Average daily study hours |
| Attendance (%) | Class attendance percentage |
| Online Courses | Number of online courses enrolled |
| Assignment Completion (%) | Percentage of assignments completed |
| Exam Score | Latest exam score out of 100 |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | XGBoost, Random Forest, Logistic Regression |
| Data Processing | pandas, NumPy, scikit-learn |
| Dashboard | Streamlit, Matplotlib |
| Explainability | Feature Importance (built-in) |
| File Format | Excel (.xlsx), Pickle (.pkl) |
