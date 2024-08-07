import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv('C:/PythonProject/diabetes.csv')


print(df.info())
print(df.describe())
print(df.head())
print(df.tail())


# Columns to check for zeros
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace zero values with NaN
df[columns_with_zeros] = df[columns_with_zeros].replace(0, np.nan)

# Check the number of NaN values in each column
print(df[columns_with_zeros].isnull().sum())

# Impute missing values using median
for column in columns_with_zeros:
    df.loc[:, column] = df[column].fillna(df[column].median())



# Verify that there are no remaining NaN values
print(df[columns_with_zeros].isnull().sum())

#grouped data by Outcomes -> Diabetic = 1 ; -> notdiabetic = 0
print(df.groupby('Outcome').mean())

#Separating data and labels
X = df.drop(columns= 'Outcome', axis = 1)
Y = df['Outcome']
print(X)
print(Y)

#Data standerdization
scaler = StandardScaler()
standerdized_data = scaler.fit_transform(X)
print(standerdized_data)

X = standerdized_data
Y = df['Outcome']
print(X)
print(Y)

#Train Test Split

X_train, X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify = Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Make predictions
y_pred = model.predict(X_test_scaled)

