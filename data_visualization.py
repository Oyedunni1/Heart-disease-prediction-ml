import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the preprocessed data
df = pd.read_csv('data/processed_data.csv')

# Function to create visualizations
def create_visualizations(df):
  
    # HeartDisease count plot
    plt.figure(figsize=(8,6))
    sns.countplot(x='HeartDisease', data=df)
    plt.title('HeartDisease Count')
    plt.savefig('visualizations/HeartDisease_count.png')
    plt.close()
    
    # Distribution of Cholesterol
    plt.figure(figsize=(8,6))
    sns.histplot(df['Cholesterol'], kde=True, bins=30)
    plt.title('Distribution of Cholesterol')
    plt.savefig('visualizations/Cholesterol_distribution.png')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()
    
    # HeartDisease vs Age
    plt.figure(figsize=(8,6))
    sns.boxplot(x='HeartDisease', y='Age', data=df)
    plt.title('HeartDisease vs Age')
    plt.savefig('visualizations/HeartDisease_vs_Age.png')
    plt.close()
    
    # HeartDisease vs Cholesterol
    plt.figure(figsize=(8,6))
    sns.boxplot(x='HeartDisease', y='Cholesterol', data=df)
    plt.title('HeartDisease vs Cholesterol')
    plt.savefig('visualizations/HeartDisease_vs_Cholesterol.png')
    plt.close()

    # HeartDisease vs ExerciseAngina
    plt.figure(figsize=(8,6))
    sns.boxplot(x='HeartDisease', y='ExerciseAngina', data=df)
    plt.title('HeartDisease vs ExerciseAngina')
    plt.savefig('visualizations/HeartDisease_vs_ExerciseAngina.png')
    plt.close()

# Create visualizations directory if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Generate visualizations
create_visualizations(df)
