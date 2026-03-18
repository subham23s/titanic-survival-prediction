# 🚢 Titanic Survival Prediction

Predicts whether a Titanic passenger survived based on features like sex, age, fare and class. Covers full ML pipeline — data cleaning, feature engineering, EDA, model training and evaluation. Built with Python and scikit-learn.

---

## 📊 Project Overview

This project works on the famous Titanic dataset (891 passengers) and solves a binary classification problem — did the passenger survive (1) or not (0)? The focus is on real-world data challenges: missing values, useless columns, and creating new meaningful features from existing ones.

Two models are trained and compared: Logistic Regression (79.33%) and Random Forest (83.24%). Feature importance analysis reveals the key factors that determined survival.

---

## 🎯 Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 79.33% |
| **Random Forest** | **83.24%** ✅ |

---

## 🔑 Key Insight — What determined survival?

| Feature | Importance |
|---------|-----------|
| Sex | 27.9% |
| Fare | 23.8% |
| Age | 23.1% |
| Passenger Class | 8.7% |
| Family Size | 5.8% |

> "Women and children first" — literally proven by the model. Sex, money and age were the three biggest survival factors on the Titanic.

---

## 🧹 Data Cleaning

- Dropped useless columns: `PassengerId`, `Name`, `Ticket`, `Cabin` (77% missing)
- Filled missing `Age` values with median (28.0)
- Filled missing `Embarked` values with mode (S)

---

## ⚙️ Feature Engineering

| New Feature | How it was created | Why |
|-------------|-------------------|-----|
| `FamilySize` | SibSp + Parch + 1 | Traveling alone vs with family |
| `IsAlone` | FamilySize == 1 | Simplified family indicator |
| `AgeGroup` | Child / Teen / Adult / Senior | Age categories |
| `FareGroup` | Low / Mid / High / Luxury | Fare categories |

---

## 🧠 Custom Predictions

```
Female, age 25, 1st class, fare $100  → SURVIVED 🟢 (88.0% confidence)
Male,   age 30, 3rd class, fare $8    → DIED     🔴 (95.3% confidence)
Female, age 8,  2nd class, fare $30   → SURVIVED 🟢 (99.0% confidence)
```

---

## 📁 Project Structure

```
titanic-survival-prediction/
│
├── titanic.py                # Main script
├── requirements.txt          # Dependencies
├── titanic_eda.png           # EDA plots (survival by sex, class, age)
├── confusion_matrix.png      # Model evaluation
├── feature_importance.png    # What factors mattered most
└── data/
    └── Titanic-Dataset.csv
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, feature engineering |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Models, metrics, encoding |

---

## 🚀 How to Run

```bash
git clone https://github.com/subham23s/titanic-survival-prediction.git
cd titanic-survival-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python titanic.py
```

---

## 🧠 What I Learned

- Handling real messy data with missing values
- Feature engineering — creating new meaningful columns
- Label encoding for categorical variables
- Confusion matrix interpretation
- How data can tell a historical story

---

## 👨‍💻 Author

**Subham Mishra**
BTech 2nd Year | CSE (AI/ML)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/subhammishra23/)


---

## 📌 Day 3 of My Daily AI/ML Build Challenge

Check out my other projects on [GitHub](https://github.com/subham23s).