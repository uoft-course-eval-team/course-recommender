import pandas as pd 
import numpy as np
from neural_net import vectorize_features, prediction, training_nn
from neural_net import NeuralNetwork
from sklearn.linear_model import LogisticRegression
import joblib
import torch
from sklearn.metrics import accuracy_score
import itertools
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import mode
"""
In this script, it creates a majority voting ensemble classifier that combines logistic regression, 
feedforward neural network with Adam optimization, and decision tree. 
"""
# Global variables 
batch_size = 10 # Batch size

def random_sampling(training_set_dataloader): 
    """Sample 3 sets from full training data"""
    # Referenced: https://stackoverflow.com/questions/47432168/taking-subsets-of-a-pytorch-dataset
    # Randomly choose 3 indicies 
    subset_1_indx = np.random.choice(len(training_set_dataloader.dataset), len(training_set_dataloader.dataset) // 3, replace = True)
    subset_2_indx = np.random.choice(len(training_set_dataloader.dataset), len(training_set_dataloader.dataset) // 3, replace = True)
    subset_3_indx = np.random.choice(len(training_set_dataloader.dataset), len(training_set_dataloader.dataset) // 3, replace = True)
    # Obtain the three sub-datasets
    subset_1 = torch.utils.data.Subset(training_set_dataloader.dataset, subset_1_indx)
    subset_2 = torch.utils.data.Subset(training_set_dataloader.dataset, subset_2_indx)
    subset_3 = torch.utils.data.Subset(training_set_dataloader.dataset, subset_3_indx)
    # Convert the datasets into Dataloaders 
    subset_1 = torch.utils.data.DataLoader(subset_1, batch_size=batch_size, shuffle=True)
    subset_2 = torch.utils.data.DataLoader(subset_2, batch_size=batch_size, shuffle=True)
    subset_3 = torch.utils.data.DataLoader(subset_3, batch_size=batch_size, shuffle=True)
    return subset_1, subset_2, subset_3

def training_all_models(models, set_1_train, set_2_train, set_3_train):
    """Training all models"""
    # All sets 
    all_train_sets = {"Neural Network": set_1_train, "Logistic Regression": set_2_train, "Decision Tree": set_3_train}
    for name, the_model in models: 
        if name != "Neural Network":
            # This model is not a neural network
            # Fit the model to the training dataset
            datapoints = [datapoint.numpy() for datapoint, actual_label in all_train_sets[name]]
            labels = [actual_label for datapoint, actual_label in all_train_sets[name]]
            x = np.vstack(datapoints)
            y = np.concatenate(labels).ravel() # Ravel function suggested by ChatGPT 
            the_model.fit(x, y)
        else: 
            # Train the model on the training set
            training_nn(nn_model, all_train_sets[name], "training", True)

def construct_ensemble(models, test_x, test_y, test_set_dataloader):
    """Construct ensemble and make prediction with majority voting"""
    # Store the predicted labels
    predicted_decision_tree = []
    predicted_logistic_reg = []
    # Make predictions for the different models
    # Make predictions and store it 
    predicted_decision_tree = models[1][1].predict(test_x)
    predicted_logistic_reg = models[0][1].predict(test_x)
    
    # For neural network
    accuracy_val, predicted_labels, actual_labels = prediction(test_set_dataloader, models[2][1])
    predicted_neural_network = np.concatenate(predicted_labels)
    
    print(predicted_decision_tree.shape)
    print(predicted_logistic_reg.shape)
    print(predicted_neural_network.shape)
    
    # Majority vote
    # Reference: https://stackoverflow.com/questions/62748195/how-to-take-the-mode-across-elements-in-multiple-numpy-arrays-of-1-d
    chosen_prediction = mode(np.stack((predicted_decision_tree, predicted_logistic_reg, predicted_neural_network), axis = 0), axis=0).mode.flatten()
    print(f'Decision Tree accuracy: {accuracy_score(test_y, predicted_decision_tree)}')
    print(f'Logistic Regression accuracy: {accuracy_score(test_y, predicted_logistic_reg)}')
    print(f'Neural Network accuracy: {accuracy_val}')
    print(f'Ensemble accuracy: {accuracy_score(test_y, chosen_prediction)}')

if __name__ == '__main__':
    # Load in datasets 
    # Read in CSV file 
    the_dataset = pd.read_csv("data/clean_data/new_data.csv")
    # Vectorize the features in the dataset
    vectorized_dataset, ct_features = vectorize_features(the_dataset)
    # print(vectorized_dataset)
    print(vectorized_dataset.shape)
    # Dataset containing the targets and turn into numpy array (so we can later turn it into a tensor)
    target_dataset = np.array(the_dataset["recommended"])
    print(target_dataset.shape)
    # Split dataset into training set and test set
    train_x, test_x, train_y, test_y = train_test_split(vectorized_dataset, target_dataset, train_size = 0.8,
                                                                              test_size=0.2, random_state=100)
    # Convert to dataloaders
    training_set_tensor = TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32).unsqueeze(1))
    test_set_tensor = TensorDataset(torch.tensor(test_x, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32).unsqueeze(1))
    # Initialize dataloaders 
    training_set_dataloader = DataLoader(dataset=training_set_tensor, batch_size=batch_size, shuffle=True)
    test_set_dataloader = DataLoader(dataset=test_set_tensor, batch_size=batch_size, shuffle=True)
    
    # # Obtain training set samples for each baseline model
    # set_1_train, set_2_train, set_3_train = random_sampling(training_set_dataloader)

    # Load in models 
    nn_model = joblib.load("models/saved_models/neural_network.sav")
    log_reg_model = joblib.load("models/saved_models/logistic_regression.sav")
    decision_tree_model = joblib.load("models/saved_models/decision_tree.sav")
    models = [("Logistic Regression", log_reg_model), ("Decision Tree", decision_tree_model), ("Neural Network", nn_model)]

    # Train models on datasets
    # training_all_models(models, set_1_train, set_2_train, set_3_train)
    log_reg_model.fit(train_x, train_y)
    log_reg_model.fit(train_x, train_y)
    training_nn(nn_model, training_set_dataloader, "training", True)
    
    # Create ensemble and obtain the majority vote  
    construct_ensemble(models, test_x, test_y, test_set_dataloader)
    
