# =============================================================
# 🚢 TITANIC SURVIVAL PREDICTION
# =============================================================
# WHAT THIS PROJECT DOES:
# Predicts whether a passenger survived the Titanic disaster
# based on features like age, sex, class, fare etc.
# This is a CLASSIFICATION problem (survived: 1 or 0)
#
# WHAT MAKES THIS SPECIAL:
# - Real messy data with missing values
# - Feature engineering (creating new features)
# - Understanding WHY people survived (story behind data)
# =============================================================


# ===== IMPORTS =====
import pandas as pd               # data manipulation
import numpy as np                # numerical operations
import matplotlib.pyplot as plt   # plotting
import seaborn as sns             # beautiful visualizations

from sklearn.model_selection import train_test_split      # split data
from sklearn.linear_model import LogisticRegression       # model 1
from sklearn.ensemble import RandomForestClassifier       # model 2
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder            # convert text to numbers
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported!")


# =============================================================
# STEP 1 — LOAD AND EXPLORE DATA
# =============================================================

df = pd.read_csv('data/Titanic-Dataset.csv')

print("\n📊 Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())


# =============================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================
# Before cleaning, let's understand the data visually
# Who survived? What patterns exist?
# =============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Overall survival count
sns.countplot(data=df, x='Survived', ax=axes[0,0], palette='Set2')
axes[0,0].set_title('Overall Survival (0=Died, 1=Survived)')
axes[0,0].set_xticklabels(['Died', 'Survived'])

# Plot 2: Survival by Sex
# Famous insight: women were prioritized ("women and children first")
sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[0,1], palette='Set2')
axes[0,1].set_title('Survival by Sex')
axes[0,1].legend(['Died', 'Survived'])

# Plot 3: Survival by Passenger Class
# 1st class = rich, 3rd class = poor
# Did money help you survive?
sns.countplot(data=df, x='Pclass', hue='Survived', ax=axes[1,0], palette='Set2')
axes[1,0].set_title('Survival by Passenger Class')
axes[1,0].legend(['Died', 'Survived'])

# Plot 4: Age distribution by survival
# Were children more likely to survive?
df[df['Survived']==1]['Age'].dropna().hist(ax=axes[1,1], alpha=0.7, label='Survived', color='green', bins=20)
df[df['Survived']==0]['Age'].dropna().hist(ax=axes[1,1], alpha=0.7, label='Died', color='red', bins=20)
axes[1,1].set_title('Age Distribution by Survival')
axes[1,1].set_xlabel('Age')
axes[1,1].legend()

plt.suptitle('Titanic EDA — Who Survived?', fontsize=14)
plt.tight_layout()
plt.savefig('titanic_eda.png')
plt.show()
print("\n✅ EDA plot saved: titanic_eda.png")


# =============================================================
# STEP 3 — DATA CLEANING
# =============================================================
# Real world data is messy. We need to handle:
# 1. Missing values (NaN)
# 2. Useless columns
# 3. Text columns (ML models need numbers)
# =============================================================

print("\n🔧 Cleaning data...")

# --- Drop columns we don't need ---
# PassengerId = just an ID number, useless for prediction
# Name = too unique to be useful (we'll extract title separately)
# Ticket = random ticket numbers, no pattern
# Cabin = too many missing values (77% missing!)
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# --- Fill missing Age values ---
# Strategy: fill with MEDIAN age (better than mean, not affected by outliers)
# Why not drop rows? Because Age is important and we'd lose too much data
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(f"   Filled missing Age with median: {median_age}")

# --- Fill missing Embarked values ---
# Embarked = port where passenger boarded (S, C, Q)
# Only 2 missing — fill with most common value (mode)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
print(f"   Filled missing Embarked with mode: {df['Embarked'].mode()[0]}")

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# =============================================================
# STEP 4 — FEATURE ENGINEERING
# =============================================================
# Feature engineering = creating NEW columns from existing ones
# This is one of the most important skills in real ML work!
# Good features can dramatically improve model performance
# =============================================================

print("\n⚙️ Engineering new features...")

# --- Feature 1: FamilySize ---
# SibSp = siblings + spouses aboard
# Parch = parents + children aboard
# Insight: traveling alone vs with family might affect survival
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 for the person themselves
print("   Created: FamilySize")

# --- Feature 2: IsAlone ---
# Simpler version of FamilySize — just alone or not
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)  # 1 if alone, 0 if with family
print("   Created: IsAlone")

# --- Feature 3: AgeGroup ---
# Instead of raw age, group into meaningful categories
# Child, Teen, Adult, Senior — each might have different survival rates
def age_group(age):
    if age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teen'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Senior'

df['AgeGroup'] = df['Age'].apply(age_group)
print("   Created: AgeGroup")

# --- Feature 4: FareGroup ---
# Group fare into categories (cheap, normal, expensive, luxury)
# Using pd.qcut which splits into equal-sized groups
df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Mid', 'High', 'Luxury'])
print("   Created: FareGroup")

print("\nNew features added:")
print(df[['FamilySize', 'IsAlone', 'AgeGroup', 'FareGroup']].head())


# =============================================================
# STEP 5 — ENCODE CATEGORICAL VARIABLES
# =============================================================
# ML models only understand numbers, not text
# Sex: male/female → 0/1
# Embarked: S/C/Q → 0/1/2
# AgeGroup, FareGroup → numbers
# =============================================================

le = LabelEncoder()

df['Sex']       = le.fit_transform(df['Sex'])        # male=1, female=0
df['Embarked']  = le.fit_transform(df['Embarked'])   # S/C/Q → 0/1/2
df['AgeGroup']  = le.fit_transform(df['AgeGroup'])   # Child/Teen/Adult/Senior → 0/1/2/3
df['FareGroup'] = le.fit_transform(df['FareGroup'])  # Low/Mid/High/Luxury → 0/1/2/3

print("\n✅ Categorical variables encoded!")
print(df.head())


# =============================================================
# STEP 6 — PREPARE DATA FOR TRAINING
# =============================================================

# Drop SibSp and Parch since we already captured them in FamilySize
X = df.drop(['Survived', 'SibSp', 'Parch'], axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"\nFeatures used: {X.columns.tolist()}")


# =============================================================
# STEP 7 — TRAIN MODEL 1: LOGISTIC REGRESSION
# =============================================================

print("\n📐 Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc  = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_acc*100:.2f}%")


# =============================================================
# STEP 8 — TRAIN MODEL 2: RANDOM FOREST
# =============================================================

print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy:       {rf_acc*100:.2f}%")


# =============================================================
# STEP 9 — EVALUATE BEST MODEL
# =============================================================

print("\n📊 Detailed Report (Random Forest):")
print(classification_report(y_test, rf_pred, target_names=['Died', 'Survived']))

# Confusion Matrix — shows exactly where the model is right/wrong
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'])
plt.title('Confusion Matrix — Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("✅ Confusion matrix saved!")


# =============================================================
# STEP 10 — FEATURE IMPORTANCE
# =============================================================

feature_imp = pd.DataFrame({
    'Feature'   : X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n🔑 Feature Importance:")
print(feature_imp)

plt.figure(figsize=(8, 5))
sns.barplot(data=feature_imp, x='Importance', y='Feature', palette='viridis')
plt.title('What factors determined survival on the Titanic?')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("✅ Feature importance saved!")


# =============================================================
# STEP 11 — PREDICT ON CUSTOM PASSENGER
# =============================================================
# Let's predict survival for a made-up passenger!
# Feature order: Pclass, Sex, Age, Fare, Embarked,
#                FamilySize, IsAlone, AgeGroup, FareGroup
# =============================================================

def predict_survival(pclass, sex, age, fare, embarked, familysize):
    is_alone   = 1 if familysize == 1 else 0
    age_group  = 0 if age <= 12 else (1 if age <= 18 else (2 if age <= 60 else 3))
    fare_group = 0 if fare < 8 else (1 if fare < 15 else (2 if fare < 31 else 3))
    sex_enc    = 1 if sex == 'male' else 0
    emb_enc    = 2 if embarked == 'S' else (0 if embarked == 'C' else 1)

    passenger  = np.array([[pclass, sex_enc, age, fare, emb_enc,
                             familysize, is_alone, age_group, fare_group]])
    prediction = rf.predict(passenger)[0]
    probability = rf.predict_proba(passenger)[0]

    result = "SURVIVED 🟢" if prediction == 1 else "DIED 🔴"
    prob   = probability[1] if prediction == 1 else probability[0]
    print(f"\nPassenger: {sex}, age {age}, class {pclass}, fare ${fare}")
    print(f"Prediction: {result}")
    print(f"Confidence: {prob*100:.1f}%")

# Test different passengers
predict_survival(pclass=1, sex='female', age=25, fare=100, embarked='S', familysize=1)
predict_survival(pclass=3, sex='male',   age=30, fare=8,   embarked='S', familysize=1)
predict_survival(pclass=2, sex='female', age=8,  fare=30,  embarked='C', familysize=3)

print("\n🎉 Project Complete!")