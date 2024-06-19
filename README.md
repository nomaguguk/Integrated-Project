<h2>Import Libraries</h2>

import pandas as pd #data preprocessing
import numpy as np
import itertools
import string

#for cross validation and model selection
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score

#base models
##linear and kernel based methods
from sklearn.linear_model import LogisticRegression

##tree-based models 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


##for deep neural networks
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers.legacy import Adam, RMSprop
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics 
from openpyxl import load_workbook
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE  # Importing SMOTE directly instead of RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns

##tensor networks
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#for preprocessing data
from sklearn.preprocessing import OneHotEncoder


##metrics 
#metrics for model evaluation and validation
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, recall_score, precision_score
from sklearn.metrics import balanced_accuracy_score, auc, roc_curve, precision_recall_curve, log_loss, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import balanced_accuracy_score
from openpyxl import load_workbook
import time
%matplotlib inline
#To show all the rows of pandas dataframe
pd.set_option('display.max_rows', None)

#for hyperparameter optimization
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import train_test_split

#model explainabilty
import shap

import os 
import time

import warnings
warnings.filterwarnings('ignore')

!pip install --upgrade openpyxl

<h2>Load the Data</h2>

# Load the CSV dataset
file_path = r"C:\Users\f5671116\OneDrive - FRG\Downloads\datasets\insurance_claims_sample.csv"
data = pd.read_csv(file_path)

print(data.head())
print(data.columns)

print('The training dataset has {} samples and {} features.'.format(data.shape[0], data.shape[1]-1))

# Display the first few rows
data.head()

<h2>EDA on Insurance Claims Data</h2>

<h2>Data Cleaning</h2>

print(data.head())

# Remove the _c39 column
data.drop(columns=['_c39'], inplace=True)

# Fill missing values
# Numerical columns: Use median
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    data[col].fillna(data[col].median(), inplace=True)

# Categorical columns: Use mode
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

data.isnull().sum()

# Verify no missing values remain
print(data.isnull().sum())

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

print(data.columns)

# Split the data into features and target
X = data.drop(columns=['fraud_reported_Y'])
y = data['fraud_reported_Y'].apply(lambda x: 1 if x == 'Y' else 0)

print(data.info())
print(data.describe(include='all'))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


data.info()

data.isna().sum().sum()

data.head(10)

data.describe()

data.age.describe()

# Display the first few rows of X_train and y_train to verify the split
print(X_train.head())
print(y_train.head())

def convert_datetime_features(data):
    for col in data.columns:
        if np.issubdtype(data[col].dtype, np.datetime64):
            data[col] = data[col].astype('int64') / 10**9  # Convert to seconds since epoch
    return data

# Apply the function to the train and test sets
X_train = convert_datetime_features(X_train)
X_test = convert_datetime_features(X_test)



def check_and_convert_types(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = data[col].astype(float)
            except ValueError:
                print(f"Column {col} cannot be converted to float directly.")
    return data

X_train = check_and_convert_types(X_train)
X_test = check_and_convert_types(X_test)


X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)


# Check class distribution in y_train
print(y_train.value_counts())

<h2>Baseline Model</h2>

!pip install --upgrade scikit-learn imbalanced-learn

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import time

# Instantiate the SMOTE oversampler
oversampler = SMOTE(random_state=42)

# Resample the training data to handle imbalance
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Instantiate the Logistic Regression classifier
base_clf = LogisticRegression(max_iter=1000, penalty='l1', solver='saga', class_weight='balanced')

# Fit the model on resampled data
start = time.time()
base_clf.fit(X_train_resampled, y_train_resampled)
end = time.time()

# Predict on the test set
y_pred = base_clf.predict(X_test)

# Calculate balanced accuracy
scores = balanced_accuracy_score(y_test, y_pred)

# Print results
print(f"Balanced Accuracy: {scores:.2f}")
print(f"Time taken: {end - start:.2f}s")

cm = 100*confusion_matrix(y_test, y_pred, normalize= 'true')
# cm = 100*cm.astype('float')/cm.sum()
fig, ax = plt.subplots(figsize = (10,10))
sns.heatmap(cm, annot = True, fmt = '.2f', xticklabels= base_clf.classes_, yticklabels= base_clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show(block=False)

print(f"-------------------------{name}---------------------------------")
print("---------------------Classification Report-----------------------")
print(classification_report(y_test, y_pred))
print("-------------------------Ensemble Method------------------------")
print('Balanced Accruacy:', scores)
print("-------------------------Run Time------------------------")
print('Total Run Time:',end)
print("------------------------Confusion Matrix--------------------")

<h2>Hyperameter Tuning using Bayesian Optimisation<h2> 

#calculate class weights 
y_train_array = np.array(insurance_df['Default'])

#calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes= np.unique(y_train), y = y_train)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
