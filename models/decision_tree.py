import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, RocCurveDisplay, precision_score, f1_score,
                             recall_score)
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from neural_net import vectorize_features, max_features_num

"""
The purpose of this file is the implementation and visualization of Decision Tree Model

Referenced: https://www.w3schools.com/python/python_ml_decision_tree.asp
Referenced: https://scikit-learn.org/stable/modules/tree.html
Referenced: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
"""

# Reading data
project_root = Path(__file__).resolve().parent.parent
csv_path = project_root /'data' /'clean_data' /'new_data.csv'

cs_data = pd.read_csv(csv_path)


# Splitting and defining data
X_encoded, col_features = vectorize_features(cs_data)
y = cs_data['recommended']

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=137)


# Building + Training model
def treefitter(X, y):
    """
    This function implements the Decision Tree Classifier. The criterion used is the default gini coefficient.
    A max depth hyperparameter of 13 was used as determined by log_2(6000) which is a heuristic value based
    on the number of rows in our feature matrix.
    :param X: The training set of features
    :param y: The target column vector
    :return: the fit random forest
    """
    model = DecisionTreeClassifier(max_depth=13, random_state=137)
    model.fit(X, y)
    return model

# fit model (why did I make this a function?)
model_final = treefitter(X_train, y_train)

#baseline model
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
baseline_accuracy = dummy.score(X_test, y_test)

# Creating model metrics
y_pred = model_final.predict(X_test)
train_pred = model_final.predict(X_train)
y_proba = model_final.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)

con_mat = confusion_matrix(y_test, y_pred)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1_score]
})


print("Train Accuracy:", accuracy_score(y_train, train_pred))
print("Test Accuracy:", accuracy)
print(con_mat)

print(f"Accuracy: {accuracy:.4f}")
print(f"Baseline accuracy: {baseline_accuracy:.4f}")
print(project_root)


# Creating Visualization

#confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(con_mat, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
save_path_cm = project_root / 'graphs' / 'decision_tree_confusion_matrix.png'
plt.savefig(save_path_cm)

#metrics table
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')
table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
save_path_df = project_root / 'graphs' / 'decision_tree_table.png'
plt.savefig(save_path_df)

#ROC curve
RocCurveDisplay.from_estimator(model_final, X_test, y_test)
plt.title("ROC Curve")
save_path_roc = project_root / 'graphs' / 'decision_tree_roc.png'
plt.savefig(save_path_roc)


