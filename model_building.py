import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
import joblib
import os

# Load the preprocessed data
df = pd.read_csv('data/processed_data.csv')

# Split the data into features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=30)

# Function to train models
def train_models(X_train, y_train):
    models = {}

    #XGBClassifier
    xgb = XGBClassifier(n_estimators = 180, learning_rate = 0.1)
    xgb.fit(X_train, y_train)
    models['XGBClassifier'] = xgb

    #KNeighborsClassifier
    knc = KNeighborsClassifier(weights = 'distance')
    knc.fit(X_train, y_train)
    models['KNeighborsClassifiers'] = knc

    #SupportVectorClassifier
    svc = SVC(kernel = 'rbf')
    svc.fit(X_train, y_train)
    models['SVC'] = svc

    #GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    models['GaussianNB'] = gnb

    #Perceptron
    pct = Perceptron(max_iter=1000)
    pct.fit(X_train, y_train)
    models['Perceptron'] = pct

    return models

# Train the models
models = train_models(X_train, y_train)

# Function to evaluate models
def evaluate_models(models, X_test, y_test):
    if not os.path.exists('model_reports'):
        os.makedirs('model_reports')
    if not os.path.exists('models'):
        os.makedirs('models')
    reports = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        reports[name] = {'report': report, 'confusion_matrix': cm}
        
        # Save the classification report
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(f'model_reports/{name}_classification_report.csv', index=True)
        
        # Save the confusion matrix
        df_cm = pd.DataFrame(cm, index=['Actual_No', 'Actual_Yes'], columns=['Predicted_No', 'Predicted_Yes'])
        df_cm.to_csv(f'model_reports/{name}_confusion_matrix.csv', index=True)
        
        # Save the model
        joblib.dump(model, f'models/{name}_model.pkl')
    
    return reports

# Evaluate the models
reports = evaluate_models(models, X_test, y_test)
