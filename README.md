📘 End-to-End Insurance Risk Analytics & Predictive Modeling

This project builds a complete data science workflow for insurance risk analysis, including data versioning, exploratory analysis, hypothesis testing, and initial predictive modeling. The goal is to help insurers identify low-risk customer segments, improve pricing strategies, and support data-driven decision making.

🔍 Key Features

Reproducible pipeline with DVC (data versioning, pipeline stages, remote storage)

Exploratory Data Analysis (EDA) with geographic, demographic, and correlation insights

Hypothesis testing & A/B experiments for understanding risk drivers

Predictive modeling using Linear Regression, Random Forest, and XGBoost

Explainability insights (SHAP)

Actionable recommendations for pricing, marketing, and data improvements

📂 Project Structure
data/          → raw & processed datasets (tracked with DVC)
src/           → preprocessing, modeling, and utility scripts
figures/       → plots from EDA & models
tables/        → summary statistics & test results
models/        → saved pipelines and model artifacts
dvc.yaml       → pipeline definition

🚀 How to Reproduce
dvc pull        # fetch datasets
dvc repro       # run full analytics pipeline

📈 Summary of Findings

Geography (province & postal code) is the strongest risk factor

Gender does not influence claim severity

Current severity models underperform; two-stage modeling recommended

Marketing should target high-margin postal codes