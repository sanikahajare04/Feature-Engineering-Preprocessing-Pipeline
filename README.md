# Feature Engineering Pipeline - Sanika Mahadev Hajare

Leak-free pipeline with ColumnTransformer + Model Tuning + Deep Learning NLP

[![Python](https://img.shields.io/badge/Python-3.13-blue)]()
[![Sklearn](https://img.shields.io/badge/Sklearn-1.9-orange)]()
[![Rabtech](https://img.shields.io/badge/Rabtech-Ready-green)]()

## 📁 Files
- feature_engineering_pipeline.ipynb - Task 1 Leak-Free FE
- model_training_tuning.ipynb - Task 2 Tuning + 4 Models
- deep_learning_nlp_classifier.ipynb - Task 3 DL NLP
- SVM_champion.pkl (AUC 0.506) - Champion
- roc_auc_curves.png + training_curves.png - Proofs

## Task 1: Leak-Free
Numeric: Imputer(median)+Scaler, Categorical: Imputer(most_frequent)+OneHot(handle_unknown=ignore), ColumnTransformer, stratify, fit only train.

## Task 2: Tuning
LogReg, RF, GB, SVM + GridSearchCV cv=3 scoring=roc_auc, Champion SVM C=10

## Task 3: DL NLP
TF-IDF 5000 + Dense 256-128-64 + BatchNorm + Dropout, POS 0.95 NEG 0.12

## Author
Sanika Mahadev Hajare - Rabtech
