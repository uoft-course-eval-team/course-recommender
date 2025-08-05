import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

"""
The purpose of this file is the implementation and visualization of Decision Tree Model
"""

# Reading data
project_root = Path(__file__).resolve().parent.parent
csv_path = project_root /'data' /'clean_data' /'new_data.csv'

cs_data = pd.read_csv(csv_path)


# Processing and splitting data
selected_features_cat = ['course', 'term', 'last name']
selected_features_num = ['item 1 (i found the course intellectually stimulating)',
                         'item 2 (the course provided me with a deep understanding of the subject manner)',
                         'item 3 (the instructor created a course atmosphere that was condusive to my learning)',
                         'item 4 (course projects, assignments, '
                         'tests, and/or exams improved my understanding of the course material)',
                         'item 5 (course projects, assignments, tests, '
                         'and/or exams provided opportunity for me to demonstrate an understanding of '
                         'the course material)',
                         'item 6 (overall, the quality of my learning experience in the course was:)',
                         'instructor generated enthusiasm',
                         'course workload']
full_features =selected_features_num + selected_features_cat + ['year']



X = cs_data[selected_features_cat]
X_encoded = pd.get_dummies(X)
Z = cs_data[selected_features_num]
F = cs_data[full_features]
y = cs_data['recommended']
g = cs_data['recommended']
h = cs_data['recommended']

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=137)
Z_train, Z_test, g_train, g_test = train_test_split(Z, g, test_size=0.2, random_state=137)
F_train, F_test, h_train, h_test = train_test_split(F, h, test_size=0.2, random_state=137)


# Building + Training model

#categorical model
cat_model = DecisionTreeClassifier(max_depth=5, random_state=137) #criterion = 'gini' is default, used for all models
cat_model.fit(X_train, y_train)

#numerical model
nodel = DecisionTreeClassifier(max_depth=5, random_state=137)
nodel.fit(Z_train, g_train)

#full model
full_model = DecisionTreeClassifier(max_depth=5, random_state=137)
full_model.fit(F_train, h_train)

#baseline model
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
baseline_accuracy = dummy.score(X_test, y_test)

# Testing model(s) and metrics
y_pred = cat_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
y_proba = cat_model.predict_proba(X_test)
g_pred = nodel.predict(Z_test)
accuracy_nodel = accuracy_score(g_test, g_pred)
h_pred = nodel.predict(F_test)
accuracy_full = accuracy_score(h_test, h_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy of nodel: {accuracy_nodel:.4f}")
print(f"Baseline accuracy: {baseline_accuracy:.4f}")
print(f"Accuracy of full: {accuracy_full:.4f}")


# Creating Visualization