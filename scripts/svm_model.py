import os
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV
import time
import joblib
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils.evaluate_model import evaluate_model

# %%
X_train = pd.read_csv('../data/train_features.csv')
X_test = pd.read_csv('../data/test_features.csv')

# %%
# X_train

# %%
# X_test

# %%
y_train = pd.read_csv('../data/train_labels.csv').squeeze()
y_test = pd.read_csv('../data/test_labels.csv').squeeze()

# %%
# y_train

# %%
# y_test

# %%
model = SVC(
    probability=True,  # Essential for .predict_proba() and roc_auc_score
    random_state=42
)

# %%
print("Training SVM model...")
model.fit(X_train, y_train)
print("Training complete.")

# %%
y_pred_class = model.predict(X_test)

# %%
# y_pred_class

# %%
# y_pred_class

# %%
y_pred_proba = model.predict_proba(X_test)[:, 1]

# %%
# y_pred_proba

# %%
result = evaluate_model(model, X_test, y_test, "SVM", False)
print("\n--- Baseline Model Evaluation ---")
print(result)

# %%
# y_train[y_train == 0].count() / y_train.count()

# %%
# y_train[y_train == 1].count() / y_train.count()

# %%
neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
scale_pos_weight_value = neg_count / pos_count
print(f"\nTraining label distribution (0/1): {neg_count/len(y_train):.2f} / {pos_count/len(y_train):.2f}")
print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

# %%
param_grid = {
    # C: Regularization parameter. Smaller C means stronger regularization.
    'C': [0.1, 1.0, 10], 
    
    # gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. 
    # Small gamma means a large radius of influence (smoother boundary).
    'gamma': [0.01, 0.1, 'scale'], 
    
    # kernel: Radial Basis Function is standard for non-linear classification.
    'kernel': ['rbf'], 
    
    # class_weight: Crucial for handling class imbalance.
    'class_weight': ['balanced']
}

# %%
start_time = time.time()
print("\nStarting GridSearchCV...")

gscv = GridSearchCV(
    # Base Estimator: XGBoost Classifier (with fixed parameters)
    estimator=SVC(
        probability=True,  # Essential for .predict_proba() and roc_auc_score
        random_state=42
    ),
    
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
best_svm_model = gscv.best_estimator_

# %% [markdown]
# Testing the best fit model on test data

# %%
y_test_proba = best_svm_model.predict_proba(X_test)[:, 1]
y_test_pred_class = best_svm_model.predict(X_test)

# %%
print('\nBest SVM Model Evaluation on Test Set (Hyperparameter Tuned):')
result = evaluate_model(best_svm_model, X_test, y_test, "SVM", True)
print(result)

# %% [markdown]
# ### Saving the model

# %%
try:
    os.makedirs('../models', exist_ok=True)
except Exception as e:
    print(f"An error occurred: {e}")

# %%
joblib.dump(best_svm_model, '../models/best_svm_model.joblib')
print("\nBest SVM model saved to '../models/best_svm_model.joblib'")