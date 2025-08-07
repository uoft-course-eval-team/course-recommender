import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split


class GaussianDiscriminantAnalysis:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.numerical_features = [
            "year",
            "item 1 (i found the course intellectually stimulating)",
            "item 2 (the course provided me with a deep understanding of the subject manner)",
            "item 3 (the instructor created a course atmosphere that was condusive to my learning)",
            "item 4 (course projects, assignments, tests, and/or exams improved my understanding of the course "
            "material)",
            "item 5 (course projects, assignments, tests, and/or exams provided opportunity for me to demonstrate an "
            "understanding of the course material)",
            "item 6 (overall, the quality of my learning experience in the course was:)",
            "instructor generated enthusiasm",
            "course workload"
        ]
        self.categorical_features = ["course", "term", "last name"]
        self.model_trained = False

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip().str.lower()
        df = df.dropna(subset=self.numerical_features + ["recommended"])
        df[self.categorical_features] = df[self.categorical_features].fillna("missing")

        X_cat = pd.get_dummies(df[self.categorical_features], drop_first=False).reset_index(drop=True)
        X_num = df[self.numerical_features].reset_index(drop=True)
        self.X = pd.concat([X_num, X_cat], axis=1).astype(np.float32)
        self.y = df["recommended"].values

    def train(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X.values, self.y, test_size=test_size, random_state=random_state
        )
        self.phi = np.mean(self.y_train)
        self.mu_0 = np.mean(self.X_train[self.y_train == 0], axis=0)
        self.mu_1 = np.mean(self.X_train[self.y_train == 1], axis=0)

        sigma = np.cov(self.X_train.T, bias=True)
        self.sigma_inv = np.linalg.pinv(sigma)

        self.model_trained = True

    def predict(self, X):
        def log_likelihood(x, mu):
            return -0.5 * np.dot((x - mu).T, np.dot(self.sigma_inv, (x - mu)))

        predictions = []
        for x in X:
            logp0 = log_likelihood(x, self.mu_0) + np.log(1 - self.phi)
            logp1 = log_likelihood(x, self.mu_1) + np.log(self.phi)
            predictions.append(int(logp1 > logp0))
        return np.array(predictions)

    def evaluate(self):
        y_pred = self.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print("GDA Accuracy:", acc)
        print(classification_report(self.y_test, y_pred))

        ConfusionMatrixDisplay.from_predictions(self.y_test, y_pred, cmap="Purples")
        plt.title("GDA Confusion Matrix")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    gda = GaussianDiscriminantAnalysis("new_data.csv")
    gda.load_and_preprocess_data()
    gda.train()
    gda.evaluate()
