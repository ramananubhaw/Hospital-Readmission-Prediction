# %%
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
import time
import joblib

# %%
X_train = pd.read_csv('../data/train_features.csv')
X_test = pd.read_csv('../data/test_features.csv')

# %%
# X_train # Output/Display Line

# %%
# X_test # Output/Display Line

# %%
y_train = pd.read_csv('../data/train_labels.csv').squeeze()
y_test = pd.read_csv('../data/test_labels.csv').squeeze()

# %%
# y_train # Output/Display Line

# %%
# y_test # Output/Display Line

# %%
model = LogisticRegression(max_iter=1000, random_state=42)

# %%
print("Training Logistic Regression model...")
model.fit(X_train, y_train)
print("Training complete.")

# %%
y_pred_class = model.predict(X_test)

# %%
# y_pred_class # Output/Display Line

# %%
# y_pred_class # Output/Display Line

# %%
y_pred_proba = model.predict_proba(X_test)[:, 1]

# %%
# y_pred_proba # Output/Display Line

# %%
def evaluate_model(model):
    y_pred_class = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class)
    recall = recall_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred_class)
    cm_df = pd.DataFrame(
        cm,
        index=['Actual No Readmission (0)', 'Actual Readmission (1)'],
        columns=['Predicted No Readmission (0)', 'Predicted Readmission (1)']
    )
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm_df)
    print("\nInterpretation of Confusion Matrix:")
    print(f"  True Negatives (TN): {cm[0, 0]} (Correctly predicted NOT readmitted)")
    print(f"  False Positives (FP): {cm[0, 1]} (Incorrectly predicted readmitted - Resource Waste)")
    print(f"  False Negatives (FN): {cm[1, 0]} (Incorrectly predicted NOT readmitted - Missed Intervention)")
    print(f"  True Positives (TP): {cm[1, 1]} (Correctly predicted readmitted)")

# %%
evaluate_model(model)

# %%
# y_train[y_train == 0].count() / y_train.count() # Output/Display Line

# %%
# y_train[y_train == 1].count() / y_train.count() # Output/Display Line

# %%
neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
scale_pos_weight_value = neg_count / pos_count
print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

# %%
param_grid = {
    # Type of regularization to use (L1 or L2)
    'penalty': ['l1', 'l2'], 
    
    # Inverse of regularization strength (C). Smaller C means stronger regularization.
    # A wide range is good to test different levels of simplicity/complexity.
    'C': [0.001, 0.01, 0.1, 1, 10, 100], 
    
    # Class weight to handle any imbalance, even if slight
    'class_weight': [
        'balanced',  # Automatically adjusts weights inversely proportional to class frequencies
        None         # No weighting (default)
    ],
    
    # Solver that supports the L1 penalty (necessary when 'penalty' includes 'l1')
    'solver': ['liblinear'] 
}

# %%
start_time = time.time()
print("\nStarting GridSearchCV...")

gscv = GridSearchCV(
    # Base Estimator: Logistic Regression Classifier
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    
    # Search Space: The dictionary of hyperparameters
    param_grid=param_grid,
    
    # Scoring Metric: Use the AUC-ROC scorer defined above
    scoring='roc_auc',
    
    # Cross-Validation Folds: 5-fold is standard
    cv=5,
    
    # Verbosity: Shows progress during the search
    verbose=3,
    
    # Number of cores to use: -1 uses all available cores for speed
    n_jobs=-1
)

# This step trains hundreds of models based on the combinations in param_grid
gscv.fit(X_train, y_train)

end_time = time.time()
print(f"\nGrid Search completed in {(end_time - start_time) / 60:.2f} minutes.")

# %%
print("\n--- Grid Search Results ---")
print(f"Best AUC-ROC Score achieved: {gscv.best_score_:.4f}")
print("Best Hyperparameters found:")
print(gscv.best_params_)

# Retrieve the best model object
best_lr_model = gscv.best_estimator_

# %% [markdown]
# Testing the best fit model on test data

# %%
y_test_proba = best_lr_model.predict_proba(X_test)[:, 1]
y_test_pred_class = best_lr_model.predict(X_test)

# %%
print('\nBest Logistic Regression Model Evaluation on Test Set:')
evaluate_model(best_lr_model)

# %% [markdown]
# ### Saving the model

# %%
try:
    os.makedirs('../models', exist_ok=True)
except Exception as e:
    print(f"An error occurred: {e}")

# %%
joblib.dump(best_lr_model, '../models/best_lr_model.joblib')