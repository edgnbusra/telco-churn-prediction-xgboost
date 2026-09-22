# Telco Customer Churn Prediction with XGBoost

An end-to-end machine learning project predicting whether a telecom customer will churn (leave the company), using the classic Telco Customer Churn dataset.

## Project steps

1. **Data exploration & preprocessing** - handling missing values and encoding categorical variables.
2. **Model training** - a baseline XGBoost classifier.
3. **Hyperparameter tuning** - parameter search to improve model performance.
4. **Class imbalance handling** - using `scale_pos_weight` (~2.77, computed from the 4139/1495 class ratio) to improve recall on the minority (churned) class.
5. **Cross-validation** - 5-fold stratified K-Fold to check the model's stability.
6. **Threshold optimization** - finding the best decision threshold via Youden's index (ROC-AUC based).

## Results

| Stage | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) |
|---|---|---|---|---|
| Baseline XGBoost | 0.774 | 0.59 | 0.50 | 0.54 |
| Tuned hyperparameters | 0.804 | 0.66 | 0.53 | 0.59 |
| + Class imbalance (`scale_pos_weight`) | 0.746 | 0.52 | 0.80 | 0.63 |
| + Optimal threshold (Youden's index) | 0.746 | 0.52 | 0.80 | 0.63 |

- **5-fold cross-validation F1 (churn class):** 0.6311 (+/- 0.0298)
- **ROC-AUC:** 0.8463

The imbalance-corrected model trades some accuracy/precision for a much higher recall on churned customers (0.50 -> 0.80) - in a churn prediction context, catching more actual churners is usually worth more than raw accuracy, since missed churners are the customers a retention campaign never reaches.

## Tech stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib.
