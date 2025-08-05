import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/clean_data/new_data.csv")
df.columns = df.columns.str.strip().str.lower()

numerical_features = [
    "year",
    "item 1 (i found the course intellectually stimulating)",
    "item 2 (the course provided me with a deep understanding of the subject manner)",
    "item 3 (the instructor created a course atmosphere that was condusive to my learning)",
    "item 4 (course projects, assignments, tests, and/or exams improved my understanding of the course material)",
    "item 5 (course projects, assignments, tests, and/or exams provided opportunity for me to demonstrate an "
    "understanding of the course material)",
    "item 6 (overall, the quality of my learning experience in the course was:)",
    "instructor generated enthusiasm",
    "course workload"
]
categorical_features = ["course", "term", "last name"]

df = df.dropna(subset=numerical_features + ["recommended"])  # Remove missing features and targets
df[categorical_features] = df[categorical_features].fillna("missing")

X_cat = pd.get_dummies(df[categorical_features], drop_first=False).reset_index(drop=True)
X_num = df[numerical_features].reset_index(drop=True)

X = pd.concat([X_num, X_cat], axis=1).astype(np.float32)  # Combine features and convert to float32
y = df["recommended"].values

# Debug
# print("X shape:", X.shape)

# Training and testing
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

# GDA parameters
phi = np.mean(y_train)
mu_0 = np.mean(X_train[y_train == 0], axis=0)
mu_1 = np.mean(X_train[y_train == 1], axis=0)

# Debug
# print("X_train shape:", X_train.shape)

sigma = np.cov(X_train.T, bias=True)  # Shared covariance matrix (regularized for stability if needed)
sigma_inv = np.linalg.pinv(sigma)  # use pseudo-inverse for stability


def predict_gda(X):  # Prediction
    def log_likelihood(x, mu):
        return -0.5 * np.dot(np.dot((x - mu).T, sigma_inv), (x - mu))

    y_pred = []
    for x in X:
        logp0 = log_likelihood(x, mu_0) + np.log(1 - phi)
        logp1 = log_likelihood(x, mu_1) + np.log(phi)
        y_pred.append(int(logp1 > logp0))
    return np.array(y_pred)


y_pred = predict_gda(X_test)  # Predict and evaluate

# Evaluation
print("GDA Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Purples")
plt.title("GDA Confusion Matrix")
plt.tight_layout()
plt.show()
