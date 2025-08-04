import pandas as pd
import sns as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("new_data.csv")
df.columns = df.columns.str.strip().str.lower()
df["recommended"] = (df["i would recommend this course"] >= 4).astype(int)  # Binary target

numerical_cols = [  # Set features
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

# Preprocessing
df = df.dropna(subset=numerical_cols + ["recommended"])
df[categorical_cols] = df[categorical_cols].fillna("missing")
df["description"] = df["description"].fillna("")

# Encoding
X_num = df[numerical_cols].reset_index(drop=True)
X_cat = pd.get_dummies(df[categorical_cols], drop_first=True).reset_index(drop=True)
vectorizer = TfidfVectorizer(max_features=100)
X_text = pd.DataFrame(vectorizer.fit_transform(df["description"]).toarray(),
                      columns=vectorizer.get_feature_names_out()).reset_index(drop=True)

# Combine features
X = pd.concat([X_num, X_cat, X_text], axis=1)
y = df["recommended"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  # Model

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Coefficients
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_.flatten()
}).sort_values(by="Coefficient", key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=coef_df.head(20), x="Coefficient", y="Feature", palette="coolwarm")
plt.title("Top 20 Logistic Regression Coefficients")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()
