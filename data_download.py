import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Function to download dataset from Kaggle
def download_kaggle_dataset():
    kaggle_dataset = 'fedesoriano/heart-failure-prediction'  # Dataset identifier
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    os.environ['KAGGLE_USERNAME'] = 'oyedunnioyewumi'  # Replace with your Kaggle username
    os.environ['KAGGLE_KEY'] = 'f0815e900b188c07f8d2a9139e808378'  # Replace with your Kaggle API key
    subprocess.call([
        'kaggle', 'datasets', 'download', kaggle_dataset, '--unzip',
        '-p', data_dir
    ])

# Download the dataset if not already downloaded
data_file = os.path.join('data', 'heart.csv')
if not os.path.exists(data_file):
    download_kaggle_dataset()

# Load the dataset
df_raw = pd.read_csv(data_file)  # Original DataFrame
print(df_raw)

#check for null values
df_raw.isnull().sum()

# Separate categorical and numerical features
col = list(df_raw.columns)
categorical_features = []
numerical_features = []
for i in col:
    if len(df_raw[i].unique()) > 6:
        numerical_features.append(i)
    else:
        categorical_features.append(i)

print('Categorical Features :',categorical_features)
print('Numerical Features :',numerical_features)

# Encode the categorical features
le = LabelEncoder()
for i in categorical_features:
    df_raw[i] = le.fit_transform(df_raw[i])

df_raw.head(5)

# Scaling the numerical features
ss = StandardScaler()
for i in numerical_features:
    df_raw[i] = ss.fit_transform(df_raw[[i]]).round(2)

df_raw.head(5)

# Changing the variable name of the already preprocessed data
df_processed = df_raw

# Save the preprocessed data
df_processed.to_csv(os.path.join('data', 'processed_data.csv'), index=False)