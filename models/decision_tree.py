import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, RocCurveDisplay
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from neural_net import max_features_num

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

def vectorize_modded(the_dataset):
    """A modified version of vectorize_features in neural_net.This function vectorizes
     the features in <the_dataset>, but ignores description as the feature will always be correlated with course
     and has caused overfitting issues.
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    # Vectorize
    ct_features = ColumnTransformer(
        transformers = [
         ("course", TfidfVectorizer(max_features = max_features_num), "course"),
         ("term", TfidfVectorizer(max_features = max_features_num), "term"),
         ("year", MinMaxScaler(), ["year"]),
         ("item 1", MinMaxScaler(), ["item 1 (i found the course intellectually stimulating)"]),
         ("item 2", MinMaxScaler(), ["item 2 (the course provided me with a deep understanding of the subject manner)"]),
         ("item 3", MinMaxScaler(), ["item 3 (the instructor created a course atmosphere that was condusive to my learning)"]),
         ("item 4", MinMaxScaler(), ["item 4 (course projects, assignments, tests, and/or exams improved my understanding of the course material)"]),
         ("item 5", MinMaxScaler(), ["item 5 (course projects, assignments, tests, and/or exams provided opportunity for me to demonstrate an understanding of the course material)"]),
         ("item 6", MinMaxScaler(), ["item 6 (overall, the quality of my learning experience in the course was:)"]),
         ("instructor generated enthusiasm", MinMaxScaler(), ["instructor generated enthusiasm"]),
         ("course workload", MinMaxScaler(), ["course workload"]),
         ("last name", TfidfVectorizer(max_features = max_features_num), "last name"),
        # ("description", TfidfVectorizer(stop_words = 'english', max_features = max_features_num), "description"
          #)
    ],
        sparse_threshold=0.0 # Parameter suggested by ChatGPT to fix NaN issue during training
    )
    vectorized_data = ct_features.fit_transform(the_dataset)
    # Convert all nan to numbers
    # https://numpy.org/doc/2.1/reference/generated/numpy.nan_to_num.html
    vectorized_data = np.nan_to_num(vectorized_data.astype(np.float32), nan = 0.0, posinf = 1.0, neginf = 0) # Function suggested by ChatGPT to fix NaN issue during training
    return vectorized_data, ct_features



X_encoded, col_features = vectorize_modded(cs_data)
y = cs_data['recommended']

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=137)


# Building + Training model

def forestfitter(X, y):
    """
    This function implements the Decision Tree Classifier as a Random Forest Classifier. The criterion used is
    the default gini. No max number of estimators was given as performance did not significantly change with
    any one of them. A max depth hyperparameter of 13 was used as determined by log_2(6000) which is a heuristic
    value based on the number of rows in our feature matrix.
    https://scikit-learn.org/stable/modules/ensemble.html#random-forests mentions that aggressive pruning of
    random forests is less necessary as in individual trees.
    :param X: The training set of features
    :param y: The target column vector
    :return: the fit random forest
    """
    model = RandomForestClassifier(max_depth=13, random_state=137)
    model.fit(X, y)
    return model

#baseline model
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
baseline_accuracy = dummy.score(X_test, y_test)

# Testing model(s) and metrics
model_final = forestfitter(X_train, y_train)
y_pred = model_final.predict(X_test)
train_pred = model_final.predict(X_train)
accuracy = accuracy_score(y_test, y_pred)
y_proba = model_final.predict_proba(X_test)

con_mat = confusion_matrix(y_test, y_pred)


print("Train Accuracy:", accuracy_score(y_train, train_pred))
print("Test Accuracy:", accuracy)

print(f"Accuracy: {accuracy:.4f}")
print(f"Baseline accuracy: {baseline_accuracy:.4f}")

"""
# Creating Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(con_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
repo_root = os.path.dirname(os.path.abspath(__file__))  # directory of the current file
save_path = os.path.join(repo_root, "graphs", "confusion_matrix_forest.png")
plt.savefig(save_path)

RocCurveDisplay.from_estimator(model_final, X_test, y_test)
plt.title("ROC Curve")
plt.show()

"""
