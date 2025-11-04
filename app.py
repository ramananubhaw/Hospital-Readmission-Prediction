import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Comparative Analysis of Classification Models")

# --- Function to Load and Prepare Data ---
@st.cache_data
def load_and_prepare_data(filepath='results.json'):
    if not os.path.exists(filepath):
        st.error(f"Error: The file '{filepath}' was not found. Please check the path.")
        return pd.DataFrame()

    with open(filepath, 'r') as f:
        results_dict = json.load(f)

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

    df = pd.DataFrame(data_list, index=[row['Model'] for row in data_list])
    df.drop('Model', axis=1, inplace=True)
    
    # Calculate derived metrics (as done in your notebook)
    df['Total Negatives'] = df['True Negatives (TN)'] + df['False Positives (FP)']
    df['Total Positives'] = df['True Positives (TP)'] + df['False Negatives (FN)']
    df['FPR'] = df['False Positives (FP)'] / df['Total Negatives']
    df['FNR'] = df['False Negatives (FN)'] / df['Total Positives']
    df['FDR'] = df['False Positives (FP)'] / (df['True Positives (TP)'] + df['False Positives (FP)'])
    
    # Sort by ROC-AUC for consistent visualization order
    df = df.sort_values(by='ROC_AUC', ascending=False)
    
    return df

# --- Load Data ---
df = load_and_prepare_data()

# --- Streamlit Interface Logic ---
st.title("Comparative Analysis of Classification Models for Hospital Readmission Prediction")

if df.empty:
    st.stop()
    
# Display the full metric table

def highlight_max_and_set_font(s, props='background-color: lightgreen; color: black;'):
    # Create a boolean mask: True where the value is the maximum
    is_max = s == s.max()
    
    # Return the style properties for max values, and an empty string otherwise
    return [props if v else '' for v in is_max]

subset_cols = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']

st.header("1. Full Metric Table")
st.dataframe(df.style.apply(
    highlight_max_and_set_font,
    props='background-color: lightgreen; color: black;',
    axis=0,
    subset=subset_cols
))

# --- Visualization Section ---
st.header("2. Overall Performance Metrics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("2.1. ROC-AUC, F1-Score, and Accuracy")
    plot_data = df[['ROC_AUC', 'F1_Score', 'Accuracy']].copy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_data.plot(kind='bar', rot=45, ax=ax)
    ax.set_title('Comparison of Primary Classification Metrics')
    ax.set_ylabel('Metric Value')
    ax.set_xlabel('Model')
    ax.set_ylim(0.5, 0.7)
    ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--')
    st.pyplot(fig)

with col2:
    st.subheader("2.2. F1-Score and ROC-AUC Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df['F1_Score'].sort_values(ascending=False).plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('F1-Score')
    axes[0].set_ylabel('F1-Score')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylim(0.5, 0.6)
    axes[0].grid(axis='y', linestyle='--')
    
    df['ROC_AUC'].sort_values(ascending=False).plot(kind='bar', ax=axes[1], color='salmon')
    axes[1].set_title('ROC-AUC')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0.6, 0.7)
    axes[1].grid(axis='y', linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

st.header("3. Trade-offs and Ranking")

# --- Visualization 2 & 10 ---
col3, col4 = st.columns(2)

with col3:
    st.subheader("3.1. Precision vs. Recall Trade-off")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0.4, 0.65], [0.4, 0.65], linestyle='--', color='gray', alpha=0.5)
    
    sns.scatterplot(x='Recall', y='Precision', data=df, s=200, hue=df.index, style=df.index, ax=ax)
    
    ax.set_title('Precision vs. Recall Trade-off')
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_xlim(0.4, 0.6)
    ax.set_ylim(0.58, 0.62)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

with col4:
    st.subheader("3.2. Precision Ranking")
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plotting the bar chart
    df['Precision'].sort_values(ascending=False).plot(kind='bar', color=sns.color_palette("pastel"), ax=ax)
    ax.set_title('Model Precision Ranking (Highest is Best)')
    ax.set_ylabel('Precision Value')
    ax.set_xlabel('Model')
    ax.set_ylim(0.58, 0.62)
    
    # Corrected alignment: Use rotation in tick_params, and manually set alignment on tick labels
    ax.tick_params(axis='x', rotation=45) 
    # Set horizontal alignment explicitly on the tick labels after rotation
    plt.setp(ax.get_xticklabels(), ha='right')

    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

st.header("4. Confusion Matrix Counts and Breakdown")

# --- Visualization 4 (Full Width) ---
st.subheader("4.1. Stacked Bar Chart of Confusion Matrix Counts")
cm_df = df[['True Positives (TP)', 'True Negatives (TN)', 'False Positives (FP)', 'False Negatives (FN)']].copy()
cm_df.columns = ['TP', 'TN', 'FP', 'FN']
cm_df = cm_df[['TP', 'FN', 'TN', 'FP']] 

fig, ax = plt.subplots(figsize=(12, 6))
cm_df.plot(kind='bar', stacked=True, rot=45, 
           color=['#4CAF50', '#F44336', '#2196F3', '#FFC107'], # Green, Red, Blue, Yellow
           ax=ax)
ax.set_title('Confusion Matrix Counts Across Models')
ax.set_ylabel('Total Sample Count')
ax.set_xlabel('Model')
ax.legend(title='CM Component', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', linestyle='--')
plt.tight_layout()
st.pyplot(fig)

# --- Visualization 5 & 6 ---
st.subheader("4.2. Positive vs. Negative Class Prediction Breakdown")
col5, col6 = st.columns(2)

with col5:
    st.caption("Focus on the positive class (TP vs FN)")
    tp_fn_df = df[['True Positives (TP)', 'False Negatives (FN)']].copy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    tp_fn_df.plot(kind='bar', rot=45, color=['#4CAF50', '#F44336'], ax=ax)
    ax.set_title('Positive Class Predictions (TP vs. FN)')
    ax.set_ylabel('Count')
    ax.legend(title='Prediction Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

with col6:
    st.caption("Focus on the negative class (FP vs TN)")
    fp_tn_df = df[['False Positives (FP)', 'True Negatives (TN)']].copy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fp_tn_df.plot(kind='bar', rot=45, color=['#FFC107', '#2196F3'], ax=ax)
    ax.set_title('Negative Class Predictions (FP vs. TN)')
    ax.set_ylabel('Count')
    ax.legend(title='Prediction Outcome')
    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

st.header("5. Error Rate and Composition Analysis")

# --- Visualization 7 & 8 ---
col7, col8 = st.columns(2)

with col7:
    st.subheader("5.1. Error Rate Comparison (FPR vs. FNR)")
    error_rate_df = df[['FPR', 'FNR']].copy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    error_rate_df.plot(kind='bar', rot=45, color=['#FFC107', '#F44336'], ax=ax)
    ax.set_title('Comparison of Error Rates')
    ax.set_ylabel('Error Rate (Proportion)')
    ax.set_xlabel('Model')
    ax.legend(title='Error Type')
    ax.set_ylim(0.2, 0.5)
    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

with col8:
    st.subheader("5.2. Breakdown of Positive Predictions (Precision vs. FDR)")
    precision_fdr_df = df[['Precision', 'FDR']].copy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    precision_fdr_df.plot(kind='bar', stacked=True, rot=45, color=['#4CAF50', '#FFC107'], ax=ax)
    ax.set_title('Precision vs. False Discovery Rate (FDR)')
    ax.set_ylabel('Proportion (Must sum to 1.0)')
    ax.set_xlabel('Model')
    ax.legend(['Precision (TP / (TP+FP))', 'FDR (FP / (TP+FP))'], title='Metric')
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

# --- Visualization 9 (Full Width) ---
st.header("6. Metric Heatmap")
st.subheader("Overall Performance Heatmap")

heatmap_data = df[['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']].T

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis", linewidths=.5, linecolor='black', ax=ax)
ax.set_title('Heatmap of Key Performance Metrics', fontsize=14)
ax.set_ylabel('Metric')
ax.set_xlabel('Model')
ax.tick_params(axis='x', rotation=45)
plt.setp(ax.get_xticklabels(), ha='right') # Set horizontal alignment explicitly on the x-labels
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
st.pyplot(fig)