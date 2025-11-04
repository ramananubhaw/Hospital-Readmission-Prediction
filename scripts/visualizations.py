# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

# %%
filepath = '../results.json'

# %%
if not os.path.exists(filepath):
    # print(f"Error: The file '{filepath}' was not found.")
    pass
else:
    # Use a 'with' statement for clean and safe file handling
    try:
        with open(filepath, 'r') as f:
            results_dict = json.load(f)
        
        # print("Successfully loaded results into 'results_dict'.")
        # Optional: Print the keys to verify the data structure
        # print("\nKeys in the loaded dictionary (model names):")
        # print(list(results_dict.keys()))
        pass

    except json.JSONDecodeError:
        # print(f"Error: Could not decode JSON from '{filepath}'. Check file format.")
        pass
    except Exception as e:
        # print(f"An unexpected error occurred: {e}")
        pass

# %%
# results_dict

# %%
data_list = []
for model_name, metrics in results_dict.items():
    row = {'Model': model_name}
    for metric, value in metrics.items():
        if metric != 'Confusion_Matrix_Values':
            if metric == 'Recall (Sensitivity)':
                row['Recall'] = value
            else:
                row[metric] = value
        else:
            for conf_metric, conf_value in value.items():
                row[conf_metric] = conf_value
    data_list.append(row)

# %%
df = pd.DataFrame(data_list, index=[row['Model'] for row in data_list])

# %%
# df

# %%
df.drop('Model', axis=1, inplace=True)

# %%
df['Total Negatives'] = df['True Negatives (TN)'] + df['False Positives (FP)']

# %%
df['Total Positives'] = df['True Positives (TP)'] + df['False Negatives (FN)']

# %%
df['FPR'] = df['False Positives (FP)'] / df['Total Negatives'] # False Positive Rate

# %%
df['FNR'] = df['False Negatives (FN)'] / df['Total Positives'] # False Negative Rate

# %%
df['FDR'] = df['False Positives (FP)'] / (df['True Positives (TP)'] + df['False Positives (FP)']) # False Discovery Rate (1 - Precision)

# %%
# df

# %% [markdown]
# ### Visualizations

# %%
try:
    os.makedirs('../visualizations', exist_ok=True)
except Exception as e:
    # print(f"An error occurred: {e}")
    pass

# %% [markdown]
# #### 1. Multi-Metric Bar Chart (ROC-AUC, F1-Score, Accuracy)

# %%
plot_data = df[['ROC_AUC', 'F1_Score', 'Accuracy']].copy()

plt.figure(figsize=(10, 6))
plot_data.plot(kind='bar', rot=45, ax=plt.gca()) # ax=plt.gca() added for consistency
plt.title('Comparison of Primary Classification Metrics (Sorted by ROC-AUC)', fontsize=14)
plt.ylabel('Metric Value')
plt.xlabel('Model')
plt.ylim(0.5, 0.7)
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('../visualizations/viz_1.png', dpi=300, bbox_inches='tight')
# plt.show()

# %% [markdown]
# #### 2. Scatter Plot: Precision vs. Recall

# %%
plt.figure(figsize=(8, 8))
# Add 45-degree line for reference
plt.plot([0.4, 0.65], [0.4, 0.65], linestyle='--', color='gray', alpha=0.5)

# This creates the markers and handles the legend
sns.scatterplot(x='Recall', y='Precision', data=df, s=200, hue=df.index, style=df.index)

plt.title('Precision vs. Recall Trade-off', fontsize=14)
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision')
plt.xlim(0.4, 0.6)
plt.ylim(0.58, 0.62)
# This line creates the key to identify the points
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.grid(True)
plt.tight_layout()
plt.savefig('../visualizations/viz_2.png', dpi=300, bbox_inches='tight')
# plt.show()

# %% [markdown]
# #### 3. F1-Score and ROC-AUC Bar Chart

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# F1-Score Plot
df['F1_Score'].sort_values(ascending=False).plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Model F1-Score Comparison', fontsize=14)
axes[0].set_ylabel('F1-Score')
axes[0].set_xlabel('Model')
axes[0].set_ylim(0.5, 0.6)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', linestyle='--')

# ROC-AUC Plot
df['ROC_AUC'].sort_values(ascending=False).plot(kind='bar', ax=axes[1], color='salmon')
axes[1].set_title('Model ROC-AUC Comparison', fontsize=14)
axes[1].set_ylabel('ROC-AUC')
axes[1].set_xlabel('Model')
axes[1].set_ylim(0.6, 0.7)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', linestyle='--')
plt.tight_layout()

plt.savefig('../visualizations/viz_3.png', dpi=300, bbox_inches='tight')
# plt.show()

# %% [markdown]
# #### 4. Stacked Bar Chart of Confusion Matrix Counts

# %%
cm_df = df[['True Positives (TP)', 'True Negatives (TN)', 'False Positives (FP)', 'False Negatives (FN)']].copy()
cm_df.columns = ['TP', 'TN', 'FP', 'FN']
cm_df = cm_df[['TP', 'FN', 'TN', 'FP']] # Reorder for visual clarity (TP/FN are Class 1, TN/FP are Class 0)

plt.figure(figsize=(10, 6))
cm_df.plot(kind='bar', stacked=True, rot=45, 
           color=['#4CAF50', '#F44336', '#2196F3', '#FFC107'],
           ax=plt.gca())
plt.title('Confusion Matrix Counts Across Models', fontsize=14)
plt.ylabel('Total Sample Count')
plt.xlabel('Model')
plt.legend(title='CM Component', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

plt.savefig('../visualizations/viz_4.png', dpi=300, bbox_inches='tight')
# plt.show()

# %% [markdown]
# #### 5. Side-by-Side Bar Chart of True Positives (TP) and False Negatives (FN)

# %%
tp_fn_df = df[['True Positives (TP)', 'False Negatives (FN)']].copy()

plt.figure(figsize=(10, 6))
tp_fn_df.plot(kind='bar', rot=45, color=['#4CAF50', '#F44336'], ax=plt.gca())
plt.title('Comparison of Positive Class Predictions (TP vs. FN)', fontsize=14)
plt.ylabel('Count')
plt.xlabel('Model')
plt.legend(title='Prediction Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

plt.savefig('../visualizations/viz_5.png', dpi=300, bbox_inches='tight')
# plt.show()

# %% [markdown]
# #### 6. Side-by-Side Bar Chart of False Positives (FP) and True Negatives (TN)

# %%
fp_tn_df = df[['False Positives (FP)', 'True Negatives (TN)']].copy()

plt.figure(figsize=(10, 6))
fp_tn_df.plot(kind='bar', rot=45, color=['#FFC107', '#2196F3'], ax=plt.gca())
plt.title('Comparison of Negative Class Predictions (FP vs. TN)', fontsize=14)
plt.ylabel('Count')
plt.xlabel('Model')
plt.legend(title='Prediction Outcome')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

plt.savefig('../visualizations/viz_6.png', dpi=300, bbox_inches='tight')
# plt.show()

# %% [markdown]
# #### 7. Error Rate Comparison (False Positive Rate vs. False Negative Rate)

# %%
error_rate_df = df[['FPR', 'FNR']].copy()

plt.figure(figsize=(10, 6))
error_rate_df.plot(kind='bar', rot=45, color=['#FFC107', '#F44336'], ax=plt.gca())
plt.title('Comparison of Error Rates (FPR vs. FNR)', fontsize=14)
plt.ylabel('Error Rate (Proportion)')
plt.xlabel('Model')
plt.legend(title='Error Type')
plt.ylim(0.2, 0.5)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

plt.savefig('../visualizations/viz_7.png', dpi=300, bbox_inches='tight')
# plt.show()

# %% [markdown]
# #### 8. Stacked Bar Chart of Proportions within Predicted Positive Class (Precision vs. FDR)

# %%
precision_fdr_df = df[['Precision', 'FDR']].copy()

plt.figure(figsize=(10, 6))
precision_fdr_df.plot(kind='bar', stacked=True, rot=45, color=['#4CAF50', '#FFC107'], ax=plt.gca())
plt.title('Breakdown of Positive Predictions (Precision vs. FDR)', fontsize=14)
plt.ylabel('Proportion (Must sum to 1.0)')
plt.xlabel('Model')
plt.legend(['Precision (TP / (TP+FP))', 'FDR (FP / (TP+FP))'], title='Metric')
plt.ylim(0.5, 1.0)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

plt.savefig('../visualizations/viz_8.png', dpi=300, bbox_inches='tight')
# plt.show()

# %% [markdown]
# #### 9. Metric Heatmap

# %%
heatmap_data = df[['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']].T

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis", linewidths=.5, linecolor='black')
plt.title('Heatmap of Key Performance Metrics', fontsize=14)
plt.ylabel('Metric')
plt.xlabel('Model')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig('../visualizations/viz_9.png', dpi=300, bbox_inches='tight')
# plt.show()

# %% [markdown]
# #### 10. Ranked Bar Chart of Precision

# %%
plt.figure(figsize=(8, 5))
df['Precision'].sort_values(ascending=False).plot(kind='bar', color=sns.color_palette("pastel"), ax=plt.gca())
plt.title('Model Precision Ranking (Highest is Best)', fontsize=14)
plt.ylabel('Precision Value')
plt.xlabel('Model')
plt.ylim(0.58, 0.62)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

plt.savefig('../visualizations/viz_10.png', dpi=300, bbox_inches='tight')
# plt.show()

# %%