import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("new_data.csv")
df.columns = df.columns.str.strip().str.lower()
df["recommended"] = (df["i would recommend this course"] >= 4).astype(int)

numerical_cols = [  # Features
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
categorical_cols = ["course", "term", "last name"]

df = df.dropna(subset=numerical_cols + ["recommended"])
df[categorical_cols] = df[categorical_cols].fillna("missing")
X_cat = pd.get_dummies(df[categorical_cols], drop_first=True).reset_index(drop=True)
X_num = df[numerical_cols].reset_index(drop=True)

X = pd.concat([X_num, X_cat], axis=1)
y = df["recommended"].values

# Training and testing
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

# Training GDA
phi = np.mean(y_train)
mu_0 = np.mean(X_train[y_train == 0], axis=0)
mu_1 = np.mean(X_train[y_train == 1], axis=0)

sigma = np.cov(X_train.T, bias=True)
sigma_inv = np.linalg.inv(sigma)


def predict_gda(X):  # Prediction
    def log_likelihood(x, mu):
        return -0.5 * np.dot(np.dot((x - mu).T, sigma_inv), (x - mu))

    y_pred = []
    for x in X:
        logp0 = log_likelihood(x, mu_0) + np.log(1 - phi)
        logp1 = log_likelihood(x, mu_1) + np.log(phi)
        y_pred.append(int(logp1 > logp0))
    return np.array(y_pred)


y_pred = predict_gda(X_test)

# Evaluation
print("GDA Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Purples")
plt.title("GDA Confusion Matrix")
plt.show()
