import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Implementation and visualization of Decision Tree Model

# Reading data
project_root = Path(__file__).resolve().parent.parent
csv_path = project_root /'data' /'clean_data' /'new_data.csv'

cs_data = pd.read_csv(csv_path)

#Transform Target Variable
def categorize_recommendation(score: float):
    """
    Helper function to transform numerical variable into categorical variable.
    :param score: float
    :return: string
    """
    if score <= 1:
        return 'Strongly Do Not Recommend'
    elif 1< score <= 2:
        return 'Do Not Recommend'
    elif 2< score <= 3:
        return 'Neutral'
    elif 3< score <= 4:
        return 'Recommend'
    else:
        return 'Strongly Recommend'

cs_data['recommendation_level'] = cs_data['I would recommend this course'
    ].apply(categorize_recommendation)



# Initializing features and target and splitting data
selected_features = ['course_code', 'instructor_rating', 'hours_per_week']
X = data[selected_features]
y = cs_data['target_column_name']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=0.2, random_state=42)

# Building + Training model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# Testing model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")


# Creating Visualization