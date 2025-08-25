import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from ast import literal_eval

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, make_scorer, classification_report



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

df_shots_clean_dummies = pd.get_dummies(df_shots_clean, columns=['period','play_pattern_name', 'technique', 'body_part'])

# Separate target and input variables

X = df_shots_clean_dummies.drop('is_goal', axis=1)
y = df_shots_clean_dummies['is_goal']


# regression algorithm

# #verify the columns and shape
# print(df_shots_clean_dummies.shape)
# print(df_shots_clean_dummies.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred_probs = model.predict_proba(X_test)[:, 1]

# final_df = X_test.copy()
# final_df['goal_prob'] = y_pred_probs

# evaluate the model

log = log_loss(y_test, y_pred_probs)
roc = roc_auc_score(y_test, y_pred_probs)
brier = brier_score_loss(y_test, y_pred_probs)

print(f"Log Loss: {log}")
print(f"ROC AUC Score: {roc}")
print(f"Brier Score Loss: {brier}")


# grid search ----------------------------------------------------------------

# # Define the hyperparameters to search
# param_grid = {
#     'penalty': ['l2'],  # Type of regularization
#     'C': [0.01, 0.1, 1, 10, 100],          # Regularization strength (smaller = stronger regularization)
#     'solver': ['lbfgs'],               # Solver for optimization
#     'max_iter': [100, 500, 1000, 5000, 10000]           # Maximum number of iterations
# }

# base_model = LogisticRegression()



# # Set up GridSearchCV
# grid_search = GridSearchCV(
#     estimator=base_model,
#     param_grid=param_grid,
#     scoring='roc_auc',
#     cv=3,               # 3-fold cross-validation
#     verbose=1,
#     n_jobs=-1
# )          # Use all processors for parallelism

# # Fit the model
# grid_search.fit(X_train, y_train)

# # Get the best parameters and best score
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print("Best Parameters:", best_params)
# print("Best ROC AUC Score:", best_score)

# # Use the best model to make predictions
# best_model = grid_search.best_estimator_
# y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# # Evaluate the model on the test set
# roc_auc = roc_auc_score(y_test, y_pred_proba)
# print(f"Test ROC AUC Score: {roc_auc:.4f}")

# # Print classification report
# y_pred = (y_pred_proba > 0.5).astype(int)
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

####################################################### to do :
# 1 -fitting with best params

model = LogisticRegression(C=100, max_iter=10000, penalty='l2', solver='lbfgs')

model.fit(X_train, y_train)

y_pred_probs = model.predict_proba(X_test)[:, 1]

logloss_final = log_loss(y_test, y_pred_probs)
roc_auc_final = roc_auc_score(y_test, y_pred_probs)
brier_score_final = brier_score_loss(y_test, y_pred_probs)

print(f"Log Loss: {logloss_final}")
print(f"ROC AUC: {roc_auc_final}")
print(f"Brier Score: {brier_score_final}")



