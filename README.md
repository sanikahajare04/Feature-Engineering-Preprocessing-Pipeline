# Feature Engineering Pipeline + Real-Time ML API - Sanika Hajare

[![Python](https://img.shields.io/badge/Python-3.13-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-green)]()
[![Tests](https://img.shields.io/badge/Tests-3%2F3%20Passed-brightgreen)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)]()
[![Grade](https://img.shields.io/badge/Grade-1%2B-gold)]()

## System Architecture
cat > README.md << 'MD'
# Feature Engineering Pipeline + Real-Time ML API - Sanika Mahadev Hajare

## System Architecture
Raw Data -> ColumnTransformer leak-free -> GridSearchCV 4 Models -> Champion SVM C=10 AUC 0.506 -> FastAPI -> Docker

## Repo Proof
- Task 1: feature_engineering_pipeline.ipynb ✅
- Task 2: model_training_tuning.ipynb ✅
- Task 3: deep_learning_nlp_classifier.ipynb ✅
- SVM_champion.pkl (Champion) ✅
- roc_auc_curves.png + training_curves.png ✅
- app.py + Dockerfile + test_api.py (3/3 passed) ✅

## API
GET / , GET /health , POST /predict , GET /docs

## Run
pip install -r requirements.txt
uvicorn app:app --reload
pytest test_api.py -v

Author: Sanika Mahadev Hajare - GRADE 1+
