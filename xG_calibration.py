import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from ast import literal_eval

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss,accuracy_score, make_scorer, classification_report
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from matplotlib.gridspec import GridSpec

from venn_abers import VennAbersCalibrator, VennAbers

import warnings
warnings.filterwarnings('ignore')



df_shots_clean = pd.read_csv('shots_data_clean.csv')

df_shots_clean = df_shots_clean.astype({
    'period': str ,
    'possession': float,
    'start_x': float,
    'start_y': float,
    'end_x': float,
    'end_y': float,
    'play_pattern_name': str,
    'is_goal': bool,
    'duration': float,
    'distance_to_goal': float,
    'angle_to_goal': float,
    'first_time': bool,
    'technique': str,
    'body_part': str,
    'defenders_in_path': float,
    'teammates_in_path': float,
    'goalkeeper_in_path': float,
    'under_pressure': bool
})

# adding standard scaler

numeric_cols = ['start_x', 'start_y', 'end_x', 'end_y', 
                'duration', 'distance_to_goal', 'angle_to_goal',
                'defenders_in_path', 'teammates_in_path']

scaler = StandardScaler()

df_shots_clean_dummies = pd.get_dummies(df_shots_clean, columns=['period','play_pattern_name', 'technique', 'body_part'])

df_shots_clean_dummies[numeric_cols] = scaler.fit_transform(df_shots_clean_dummies[numeric_cols])

# Separate target and input variables

X = df_shots_clean_dummies.drop('is_goal', axis=1)
y = df_shots_clean_dummies['is_goal']


# regression algorithm

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred_probs = model.predict_proba(X_test)[:, 1]

# evaluate the model

# log = log_loss(y_test, y_pred_probs)
# roc = roc_auc_score(y_test, y_pred_probs)
# brier = brier_score_loss(y_test, y_pred_probs)

# print(f"Log Loss: {log}")
# print(f"ROC AUC Score: {roc}")
# print(f"Brier Score Loss: {brier}")


# apply calibration and draw calibration plots

# Original uncalibrated predictions
uncalibrated_probs = model.predict_proba(X_test)[:, 1]

# Inductive Venn-ABERS calibration (IVAP)
va_inductive = VennAbersCalibrator(estimator=model, inductive=True, cal_size=0.2, shuffle=False)
va_inductive.fit(X_train, y_train)
va_inductive_probs = va_inductive.predict_proba(X_test)

# Cross Venn-ABERS calibration (CVAP)
va_cross = VennAbersCalibrator(estimator=model, inductive=False, n_splits=2)
va_cross.fit(X_train, y_train)
va_cross_probs = va_cross.predict_proba(X_test)

# Pre-fitted Venn-ABERS calibration
X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=False, random_state=42
)

model_prefit = LogisticRegression(max_iter=1000)
model_prefit.fit(X_train_proper, y_train_proper)
p_cal = model_prefit.predict_proba(X_cal)
p_test = model_prefit.predict_proba(X_test)
y_cal = np.array(y_cal)  # Convert pandas Series to numpy array

va_prefit = VennAbersCalibrator()
va_prefit_probs = va_prefit.predict_proba(p_cal=p_cal, y_cal=y_cal, p_test=p_test)

# Compare metrics for all methods
methods = ['Uncalibrated', 'IVAP', 'CVAP', 'Prefit']
probabilities = [uncalibrated_probs, va_inductive_probs, va_cross_probs, va_prefit_probs]

print("\nCalibration Comparison:")
print("-" * 50)
for method, probs in zip(methods, probabilities):
    # Extract probabilities for positive class if 2D array
    if isinstance(probs, np.ndarray) and len(probs.shape) > 1:
        probs = probs[:, 1]
    log_loss_score = log_loss(y_test, probs)
    roc_score = roc_auc_score(y_test, probs)
    brier_score = brier_score_loss(y_test, probs)
    
    print(f"\n{method}:")
    print(f"Log Loss: {log_loss_score:.4f}")
    print(f"ROC AUC: {roc_score:.4f}")
    print(f"Brier Score: {brier_score:.4f}")

# Add calibration plots
plt.figure(figsize=(10, 10))
for method, probs in zip(methods, probabilities):
    if isinstance(probs, np.ndarray) and len(probs.shape) > 1:
        probs = probs[:, 1]
    
    CalibrationDisplay.from_predictions(
        y_test,
        probs,
        n_bins=10,
        name=method,
        label=method
    )

plt.title('Calibration Curves')
plt.grid()
plt.show()