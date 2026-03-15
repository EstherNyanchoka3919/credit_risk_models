# =============================================================================
# CREDIT RISK MODELING – PROFESSIONAL SOLUTION
# Data: application_record.csv, credit_record.csv (Kaggle)
# Author: Senior Data Scientist
# Date: 2025-03-15
# =============================================================================

# %% [markdown]
# ## 1. Business Problem
# We aim to predict the probability that a credit applicant will default (become a “bad” customer) based on their application data and past credit history.  
# This model will be used by loan officers to make faster, more consistent decisions and by risk managers to monitor portfolio quality.

# %% [markdown]
# ## 2. Environment Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Preprocessing & modeling
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (roc_auc_score, precision_recall_curve, 
                             classification_report, confusion_matrix, 
                             f1_score, recall_score, precision_score)

# Imbalance handling
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Models
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Hyperparameter tuning
import optuna
from sklearn.model_selection import RandomizedSearchCV

# Explainability
import shap

# Plotly for Streamlit (used later)
import plotly.express as px
import plotly.graph_objects as go

# %% [markdown]
# ## 3. Data Loading & Initial Exploration

app = pd.read_csv('application_record.csv')
cred = pd.read_csv('credit_record.csv')

print("Application shape:", app.shape)
print("Credit record shape:", cred.shape)

# Display basic info
app.info()
cred.info()

# Check missing values
app.isnull().sum()
cred.isnull().sum()

# %% [markdown]
# ## 4. Data Cleaning & Preprocessing

# Rename columns for clarity
app.rename(columns={
    'CODE_GENDER': 'GENDER',
    'FLAG_OWN_CAR': 'OWN_CAR',
    'FLAG_OWN_REALTY': 'OWN_REALTY',
    'CNT_CHILDREN': 'CHILDREN',
    'AMT_INCOME_TOTAL': 'ANNUAL_INCOME',
    'NAME_INCOME_TYPE': 'INCOME_TYPE',
    'NAME_EDUCATION_TYPE': 'EDUCATION',
    'NAME_FAMILY_STATUS': 'FAMILY_STATUS',
    'NAME_HOUSING_TYPE': 'HOUSING',
    'DAYS_BIRTH': 'AGE_DAYS',
    'DAYS_EMPLOYED': 'EMPLOYED_DAYS',
    'FLAG_PHONE': 'PHONE',
    'FLAG_EMAIL': 'EMAIL',
    'OCCUPATION_TYPE': 'OCCUPATION',
    'CNT_FAM_MEMBERS': 'FAMILY_SIZE'
}, inplace=True)

# Age in years
app['AGE'] = (app['AGE_DAYS'] / -365).astype(int)

# Employment years – special flag 365243 means unemployed
app['EMPLOYED_YEARS'] = app['EMPLOYED_DAYS'].apply(lambda x: 0 if x == 365243 else x / -365)
app['EMPLOYED_YEARS'] = app['EMPLOYED_YEARS'].round(1)

# Drop original day columns
app.drop(['AGE_DAYS', 'EMPLOYED_DAYS'], axis=1, inplace=True)

# Missing occupation – use KNN imputer on selected features?
# But occupation is categorical; we can impute with mode per income group
app['OCCUPATION'].fillna('Unknown', inplace=True)

# %% [markdown]
# ## 5. Feature Engineering

# 5.1 Target variable from credit_record
# Create binary target: 1 if ever had status 2,3,4,5 (>=60 days overdue)
cred['BAD'] = cred['STATUS'].apply(lambda x: 1 if x in ['2','3','4','5'] else 0)

# Aggregate per customer: maximum BAD (if ever defaulted)
cred_agg = cred.groupby('ID').agg(
    TOTAL_MONTHS=('MONTHS_BALANCE', 'count'),
    EVER_BAD=('BAD', 'max'),
    AVG_RISK=('BAD', 'mean')
).reset_index()

# Account age = max absolute months balance
cred_agg['ACCOUNT_AGE'] = cred.groupby('ID')['MONTHS_BALANCE'].apply(lambda x: x.abs().max()).values

# Merge with application data
df = pd.merge(app, cred_agg, on='ID', how='inner')
print("Merged shape:", df.shape)

# 5.2 Derived financial ratios (5 C's of Credit)
# Capacity: income per family member
df['INCOME_PER_CAPITA'] = df['ANNUAL_INCOME'] / df['FAMILY_SIZE']

# Capital: could use employment stability
df['EMPLOYMENT_STABILITY'] = df['EMPLOYED_YEARS'] / (df['AGE'] - 18 + 1).clip(lower=1)

# Condition: create interaction of age and income
df['AGE_INCOME'] = df['AGE'] * df['ANNUAL_INCOME'] / 1e6  # scaled

# 5.3 Encode categorical variables
cat_cols = ['GENDER', 'OWN_CAR', 'OWN_REALTY', 'INCOME_TYPE', 
            'EDUCATION', 'FAMILY_STATUS', 'HOUSING', 'OCCUPATION']

# One-hot encode
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Binary columns mapping
df['OWN_CAR'] = df['OWN_CAR'].map({'Y':1, 'N':0})
df['OWN_REALTY'] = df['OWN_REALTY'].map({'Y':1, 'N':0})

# 5.4 Drop leakage columns
leakage = ['ID', 'EVER_BAD', 'AVG_RISK', 'TOTAL_MONTHS', 'ACCOUNT_AGE']
X = df.drop(columns=leakage + ['BAD'])
y = df['BAD']

print("Features shape:", X.shape)
print("Target distribution:\n", y.value_counts(normalize=True))

# %% [markdown]
# **Target imbalance** – only about 1.5% bad. We will handle it with SMOTE‑ENN.

# %% [markdown]
# ## 6. Train/Test Split & Preprocessing Pipeline

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Identify numeric columns for scaling
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Create a preprocessor
preprocessor = ColumnTransformer([
    ('scaler', StandardScaler(), num_cols)
])

# %% [markdown]
# ## 7. Handling Imbalance with SMOTE‑ENN

# Build an imbalance‑handling pipeline with SMOTE‑ENN
smote_enn = SMOTEENN(random_state=42)

# %% [markdown]
# ## 8. Model Training & Hyperparameter Tuning with Optuna

# We will compare three models: Random Forest, XGBoost, LightGBM
# Use stratified 5‑fold CV to optimise ROC‑AUC

def objective_rf(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': 42,
        'n_jobs': -1
    }
    model = RandomForestClassifier(**params)
    # Use pipeline with SMOTE-ENN and model
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('sampling', smote_enn),
        ('classifier', model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

# Optimise Random Forest
study_rf = optuna.create_study(direction='maximize', study_name='RF')
study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train), n_trials=30)
print('Best RF params:', study_rf.best_params)
print('Best RF CV AUC:', study_rf.best_value)

# Similarly for XGBoost
def objective_xgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    model = xgb.XGBClassifier(**params)
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('sampling', smote_enn),
        ('classifier', model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

study_xgb = optuna.create_study(direction='maximize', study_name='XGB')
study_xgb.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=30)
print('Best XGB params:', study_xgb.best_params)
print('Best XGB CV AUC:', study_xgb.best_value)

# LightGBM
def objective_lgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }
    model = lgb.LGBMClassifier(**params, verbose=-1)
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('sampling', smote_enn),
        ('classifier', model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

study_lgb = optuna.create_study(direction='maximize', study_name='LGB')
study_lgb.optimize(lambda trial: objective_lgb(trial, X_train, y_train), n_trials=30)
print('Best LGB params:', study_lgb.best_params)
print('Best LGB CV AUC:', study_lgb.best_value)

# Choose best model based on CV AUC
best_model_name = max(
    [('RF', study_rf.best_value), ('XGB', study_xgb.best_value), ('LGB', study_lgb.best_value)],
    key=lambda x: x[1]
)[0]

print(f'Best model: {best_model_name}')

# %% [markdown]
# ## 9. Train Final Model with Best Parameters

if best_model_name == 'RF':
    best_params = study_rf.best_params
    model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
elif best_model_name == 'XGB':
    best_params = study_xgb.best_params
    model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
else:
    best_params = study_lgb.best_params
    model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)

# Final pipeline
final_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('sampling', smote_enn),
    ('classifier', model)
])

final_pipeline.fit(X_train, y_train)

# Predictions
y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.3).astype(int)  # threshold adjusted for recall

# %% [markdown]
# ## 10. Evaluation (business‑focused metrics)

print('ROC‑AUC:', roc_auc_score(y_test, y_pred_proba))
print('\nClassification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Precision‑Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8,5))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision‑Recall Curve')
plt.grid(True)
plt.show()

# %% [markdown]
# ## 11. Model Explainability with SHAP

# Extract the trained classifier from pipeline
classifier = final_pipeline.named_steps['classifier']
X_test_preprocessed = final_pipeline.named_steps['preprocessor'].transform(X_test)

# Use SHAP TreeExplainer for tree models
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_test_preprocessed)

# Summary plot
shap.summary_plot(shap_values, X_test_preprocessed, feature_names=num_cols)

# Waterfall plot for a single applicant (example)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test_preprocessed[0,:], feature_names=num_cols)

# %% [markdown]
# ## 12. Save Model & Preprocessor for Deployment

import joblib
joblib.dump(final_pipeline, 'credit_risk_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(num_cols, 'num_cols.pkl')  # feature names for SHAP

print('Model and artifacts saved.')
