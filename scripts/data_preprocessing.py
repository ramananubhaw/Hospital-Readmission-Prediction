# %%
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, MinMaxScaler

# %%
load_dotenv()

# %%
filepath = os.getenv('DATASET_PATH')

# %%
df = pd.read_csv(filepath)

# %%
# df # Output/Display Line

# %%
# df.shape # Output/Display Line

# %%
# df.columns # Output/Display Line

# %%
# Checking for missing values

# df.isnull().sum() # Output/Display Line

# %% [markdown]
# ### Splitting the dataset into training and testing sets

# %%
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# %%
# X # Output/Display Line

# %%
# y # Output/Display Line

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
# X_train.shape # Output/Display Line

# %%
# y_train.shape # Output/Display Line

# %%
# X_test.shape # Output/Display Line

# %%
# y_test.shape # Output/Display Line

# %% [markdown]
# ### Analyzing categorical features

# %%
# Analyzing age feature

# X_train['age'].unique() # Output/Display Line

# %%
# X_test['age'].unique() # Output/Display Line

# %%
encoder = OneHotEncoder()
age_encoded_train = encoder.fit_transform(X_train[['age']])
age_encoded_test = encoder.transform(X_test[['age']])

# %%
# age_encoded_train.toarray() # Output/Display Line

# %%
# age_encoded_test.toarray() # Output/Display Line

# %%
# encoder.get_feature_names_out(['age']) # Output/Display Line

# %%
age_encoded_train_df = pd.DataFrame(age_encoded_train.toarray(), columns=encoder.get_feature_names_out(['age']))

# %%
# age_encoded_train_df # Output/Display Line

# %%
age_encoded_test_df = pd.DataFrame(age_encoded_test.toarray(), columns=encoder.get_feature_names_out(['age']))

# %%
# age_encoded_test_df # Output/Display Line

# %%
X_train.drop('age', axis=1, inplace=True)

# %%
X_test.drop('age', axis=1, inplace=True)

# %%
X_train.reset_index(inplace=True)
X_test.reset_index(inplace=True)

# %%
# X_train # Output/Display Line

# %%
# X_test # Output/Display Line

# %%
X_train = pd.concat([X_train, age_encoded_train_df], axis=1)

# %%
# X_train # Output/Display Line

# %%
X_test = pd.concat([X_test, age_encoded_test_df], axis=1)

# %%
# X_test # Output/Display Line

# %%
X_train.drop('index', axis=1, inplace=True)

# %%
X_test.drop('index', axis=1, inplace=True)

# %%
# X_train # Output/Display Line

# %%
# X_test # Output/Display Line

# %%
# Analyzing medical_specialty feature

# X_train['medical_specialty'].unique() # Output/Display Line

# %%
# X_test['medical_specialty'].unique() # Output/Display Line

# %%
encoder = OneHotEncoder()
medspec_encoded_train = encoder.fit_transform(X_train[['medical_specialty']])
medspec_encoded_test = encoder.transform(X_test[['medical_specialty']])

# %%
# medspec_encoded_train.toarray() # Output/Display Line

# %%
# medspec_encoded_test.toarray() # Output/Display Line

# %%
# encoder.get_feature_names_out(['medical_specialty']) # Output/Display Line

# %%
medspec_encoded_train_df = pd.DataFrame(medspec_encoded_train.toarray(), columns=encoder.get_feature_names_out(['medical_specialty']))

# %%
# medspec_encoded_train_df # Output/Display Line

# %%
medspec_encoded_test_df = pd.DataFrame(medspec_encoded_test.toarray(), columns=encoder.get_feature_names_out(['medical_specialty']))

# %%
# medspec_encoded_test_df # Output/Display Line

# %%
X_train.drop('medical_specialty', axis=1, inplace=True)
X_test.drop('medical_specialty', axis=1, inplace=True)

# %%
X_train = pd.concat([X_train, medspec_encoded_train_df], axis=1)

# %%
X_test = pd.concat([X_test, medspec_encoded_test_df], axis=1)

# %%
# X_train # Output/Display Line

# %%
# X_test # Output/Display Line

# %%
# X_train.columns # Output/Display Line

# %%
# Analyzing diag features

# X_train['diag_1'].unique() # Output/Display Line

# %%
# X_test['diag_1'].unique() # Output/Display Line

# %%
# X_train['diag_2'].unique() # Output/Display Line

# %%
# X_test['diag_2'].unique() # Output/Display Line

# %%
# X_train['diag_3'].unique() # Output/Display Line

# %%
# X_test['diag_3'].unique() # Output/Display Line

# %%
encoder = OneHotEncoder(handle_unknown='ignore')
diag_encoded_train = encoder.fit_transform(X_train[['diag_1', 'diag_2', 'diag_3']])
diag_encoded_test = encoder.transform(X_test[['diag_1', 'diag_2', 'diag_3']])

# %%
# diag_encoded_train.toarray() # Output/Display Line

# %%
# diag_encoded_test.toarray() # Output/Display Line

# %%
# encoder.get_feature_names_out(['diag_1', 'diag_2', 'diag_3']) # Output/Display Line

# %%
diag_encoded_train_df = pd.DataFrame(diag_encoded_train.toarray(), columns=encoder.get_feature_names_out(['diag_1', 'diag_2', 'diag_3']))

# %%
# diag_encoded_train_df # Output/Display Line

# %%
diag_encoded_test_df = pd.DataFrame(diag_encoded_test.toarray(), columns=encoder.get_feature_names_out(['diag_1', 'diag_2', 'diag_3']))

# %%
# diag_encoded_test_df # Output/Display Line

# %%
X_train.drop(['diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)

# %%
X_test.drop(['diag_1', 'diag_2', 'diag_3'], axis=1, inplace=True)

# %%
X_train = pd.concat([X_train, diag_encoded_train_df], axis=1)

# %%
X_test = pd.concat([X_test, diag_encoded_test_df], axis=1)

# %%
# X_train # Output/Display Line

# %%
# X_test # Output/Display Line

# %%
# X_train.shape # Output/Display Line

# %%
# X_test.shape # Output/Display Line

# %%
# X_train.columns # Output/Display Line

# %%
# Analyzing glucose_test feature

# X_train['glucose_test'].unique() # Output/Display Line

# %%
# X_test['glucose_test'].unique() # Output/Display Line

# %%
encoder = OrdinalEncoder(categories=[['no', 'normal', 'high']])
glc_encoded_train = encoder.fit_transform(X_train[['glucose_test']])
glc_encoded_test = encoder.transform(X_test[['glucose_test']])

# %%
# glc_encoded_train # Output/Display Line

# %%
# glc_encoded_test # Output/Display Line

# %%
# encoder.categories_ # Output/Display Line

# %% [markdown]
# no - 0<br>
# normal - 1<br>
# high - 2<br>

# %%
X_train['glucose_test'] = glc_encoded_train

# %%
X_test['glucose_test'] = glc_encoded_test

# %%
# X_train # Output/Display Line

# %%
# X_train.columns # Output/Display Line

# %%
# Analyzing the A1Ctest feature

# X_train['A1Ctest'].unique() # Output/Display Line

# %%
encoder = OrdinalEncoder(categories=[['no', 'normal', 'high']])
A1C_encoded_train = encoder.fit_transform(X_train[['A1Ctest']])
A1C_encoded_test = encoder.transform(X_test[['A1Ctest']])

# %%
X_train['A1Ctest'] = A1C_encoded_train

# %%
X_test['A1Ctest'] = A1C_encoded_test

# %%
# X_train # Output/Display Line

# %%
# X_test # Output/Display Line

# %%
# X_train.columns # Output/Display Line

# %%
# X_train['change'].unique() # Output/Display Line

# %%
# X_test['change'].unique() # Output/Display Line

# %%
# X_train['diabetes_med'].unique() # Output/Display Line

# %%
# X_test['diabetes_med'].unique() # Output/Display Line

# %%
encoder = LabelEncoder()
encoded_diab_train = encoder.fit_transform(X_train[['diabetes_med']].to_numpy().ravel())
encoded_diab_test = encoder.transform(X_test[['diabetes_med']].to_numpy().ravel())

# %%
# encoded_diab_train # Output/Display Line

# %%
# encoded_diab_test # Output/Display Line

# %%
X_train['diabetes_med'] = encoded_diab_train

# %%
X_test['diabetes_med'] = encoded_diab_test

# %%
# X_train # Output/Display Line

# %%
# X_test # Output/Display Line

# %%
encoder = LabelEncoder()
encoded_change_train = encoder.fit_transform(X_train[['change']].to_numpy().ravel())
encoded_change_test = encoder.transform(X_test[['change']].to_numpy().ravel())

# %%
# encoded_change_train # Output/Display Line

# %%
# encoded_change_test # Output/Display Line

# %%
X_train['change'] = encoded_change_train

# %%
X_test['change'] = encoded_change_test

# %%
# X_train # Output/Display Line

# %%
# X_test # Output/Display Line

# %%
# X_train.columns # Output/Display Line

# %% [markdown]
# ### Analyzing numerical features

# %%
# X_train['time_in_hospital'].unique() # Output/Display Line

# %%
# X_train['n_lab_procedures'].unique() # Output/Display Line

# %%
# X_train['n_procedures'].unique() # Output/Display Line

# %%
# X_train['n_medications'].unique() # Output/Display Line

# %%
# X_train['n_outpatient'].unique() # Output/Display Line

# %%
# X_train['n_inpatient'].unique() # Output/Display Line

# %%
# X_train['n_emergency'].unique() # Output/Display Line

# %%
# Applying Min-Max Scaling to numerical features

scaler = MinMaxScaler()
numerical_features = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# %%
# X_train # Output/Display Line

# %%
# X_test # Output/Display Line

# %%
def sanitize_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[', '').str.replace(')', '')
    return df

# %%
X_train = sanitize_column_names(X_train)
X_test = sanitize_column_names(X_test)

# %%
try:
    os.makedirs('../data', exist_ok=True)
except Exception as e:
    print(f"An error occurred: {e}")

# %%
X_train.to_csv('../data/train_features.csv', index=False)
X_test.to_csv('../data/test_features.csv', index=False)

# %% [markdown]
# ### Analyzing output feature

# %%
# y_train.unique() # Output/Display Line

# %%
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# %%
# y_train_encoded # Output/Display Line

# %%
# y_test_encoded # Output/Display Line

# %%
y_train_series = pd.Series(y_train_encoded, name='readmission_status')
y_test_series = pd.Series(y_test_encoded, name='readmission_status')

# %%
y_train_series.to_csv('../data/train_labels.csv', index=False)
y_test_series.to_csv('../data/test_labels.csv', index=False)