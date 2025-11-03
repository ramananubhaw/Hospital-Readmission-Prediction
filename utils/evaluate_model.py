from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import os
import json

def evaluate_model(model, X_test, y_test, model_name, save):
    """
    Evaluates the model and returns a dictionary containing all metrics.
    
    Args:
        model: Trained scikit-learn compatible model.
        X_test: Test features DataFrame.
        y_test: True test labels Series/Array.
        model_name (str): Name of the model.
        save (bool): Whether to save the evaluation results to a file.
        
    Returns:
        dict: A dictionary of all performance metrics and confusion matrix values.
    """
    y_pred_class = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred_class)

    metrics_data = {
        "model_name": model_name,
        "Accuracy": accuracy_score(y_test, y_pred_class),
        "Precision": precision_score(y_test, y_pred_class),
        "Recall (Sensitivity)": recall_score(y_test, y_pred_class),
        "F1_Score": f1_score(y_test, y_pred_class),
        "ROC_AUC": roc_auc_score(y_test, y_pred_proba)
    }

    cm_data = {
        "True Negatives (TN)": int(cm[0, 0]),
        "False Positives (FP)": int(cm[0, 1]),
        "False Negatives (FN)": int(cm[1, 0]),
        "True Positives (TP)": int(cm[1, 1])
    }

    results = metrics_data
    results['Confusion_Matrix_Values'] = cm_data

    if save:
        # Assuming the final JSON file is in the root directory
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        RESULTS_FILE = os.path.join(ROOT_DIR, 'results.json')
        
        # --- FIX START: Check file size before loading ---
        all_results = {}
        if os.path.exists(RESULTS_FILE) and os.path.getsize(RESULTS_FILE) > 0:
            try:
                with open(RESULTS_FILE, 'r') as f:
                    all_results = json.load(f)
            except json.JSONDecodeError:
                # Handle case where the file exists but is corrupted/malformed
                print(f"Warning: Existing results file is corrupted. Starting fresh for {model_name}.")
                all_results = {}
        # --- FIX END ---
        
        # Update the dictionary with the current model's results (model_name is the key)
        all_results[model_name] = results
        
        # Write the aggregated results back to the JSON file
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print(f"Metrics for {model_name} saved/updated in {RESULTS_FILE}")

    return results