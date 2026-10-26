# Case Study 3: Logistic Regression – Titanic Survival Prediction

Fundamentals of Data Science.

## Files
| File | Description |
|---|---|
| `titanic_logistic_regression.ipynb` | Full analysis: EDA, preprocessing, model, evaluation, interpretation |
| `titanic.csv` | Dataset (891 passengers) |
| `app.py` | Streamlit app for interactive predictions |
| `titanic_model.joblib` | Trained pipeline (scaler + logistic regression) |
| `requirements.txt` | Python dependencies |

## Run
```bash
pip install -r requirements.txt
jupyter notebook titanic_logistic_regression.ipynb   # re-run the analysis
streamlit run app.py                                 # launch the app
```

## Results (test set)
Accuracy 0.821, Precision 0.814, Recall 0.696, F1 0.750, ROC-AUC 0.841.

## Deployment
Push this folder to GitHub, then deploy `app.py` on Streamlit Community Cloud.
