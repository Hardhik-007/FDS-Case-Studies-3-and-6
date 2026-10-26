# Case Study 6 — Bank Marketing Prediction

**Student:** P Hardhik Reddy  
**Roll No.:** 160124733192  
**Section:** CSE-3  
**Subject:** Fundamentals of Data Science (22CSC59)  
**Faculty:** S Durga Devi

## Objective
Compare Logistic Regression, Decision Tree, KNN, and Random Forest for predicting whether a customer subscribes to a term deposit.

## Files
- `Case_Study_6_Report.pdf` — report with methodology, findings, plots, metrics, recommendations, and limitations.
- `Case_Study_6_Bank_Marketing.ipynb` — notebook with code and executed outputs.
- `bank_marketing_analysis.py` — reproducible Python analysis script.
- `data/bank-full.csv` — supplied dataset (semicolon-delimited).
- `figures/` — saved charts and confusion matrix exports.
- `model_comparison.csv` — actual held-out test-set metrics.

## Run
From this folder:
```bash
pip install -r requirements.txt
jupyter notebook Case_Study_6_Bank_Marketing.ipynb
```
Or run `python bank_marketing_analysis.py`.

## Important modeling choice
`duration` is excluded from primary predictors because it is not known before a call begins. The train/test split is stratified (80:20, random_state=42). The target is imbalanced (11.70% subscribed), so accuracy should not be used alone.

## Current result summary
Decision Tree has the highest F1-score (0.358) and recall (0.268) at the default threshold. Random Forest has the highest ROC-AUC (0.803). Results depend on this split and configuration; threshold tuning and cross-validation are recommended.

## Academic integrity
Review the notebook, run it yourself, and ensure you understand and can explain the choices and outputs. This is a reproducible starting point; your course requires individual work.
