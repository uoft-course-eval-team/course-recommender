import pandas as pd 
import numpy as np
from neural_net import vectorize_features, split_dataset, convert_to_dataloaders, prediction, training_nn
from neural_net import NeuralNetwork
from sklearn.linear_model import LogisticRegression
import joblib
import torch
from sklearn.metrics import accuracy_score
import itertools
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
"""
In this script, it creates a weighted average ensemble classifier with bagging that combines logistic regression, 
feedforward neural network with Adam optimization, and decision tree. 
"""
def random_sampling(train_x, train_y): 
    """Sample 2 sets from full training data"""
    np.random.choice()
    set_1_x, set_2_x, set_1_y, set_2_y = train_test_split(train_x, train_y, train_size = 0.5, test_size=0.5, random_state=45)
    return set_1_x, set_2_x, set_1_y, set_2_y
    
def obtain_scores_for_base_models(models, train_x, train_y, validation_x, validation_y, training_set_dataloader, validation_set_dataloader):
    """Obtain scores for base models in <models>."""
    # Scores
    scores = []
    for name, the_model in models: 
        if name != "nn":
            # This model is not a neural network
            # Fit the model to the training dataset
            the_model.fit(train_x, train_y)
            # Make prediction 
            predicted_label = the_model.predict(validation_x)
            scores += [accuracy_score(validation_y, predicted_label)]
        else: 
            # Train the model on the training set
            training_nn(nn_model, training_set_dataloader, "training", True)
            # Make prediction
            accuracy_val, predicted_labels, actual_labels = prediction(validation_set_dataloader, nn_model)
            scores += [accuracy_val]
    return scores
                    

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
    # Split dataset into training set, validation set, and test set
    train_x, train_y, validation_x, validation_y, test_x, test_y = split_dataset(vectorized_dataset, target_dataset)
    training_set_dataloader, validation_set_dataloader, test_set_dataloader = convert_to_dataloaders(train_x, train_y, validation_x, validation_y, test_x, test_y)
    # Obtain training set samples for each baseline model
    set_1_x, set_2_x, set_1_y, set_2_y = random_sampling(train_x, train_y)
    # Convert one of the training set to dataloader 
    set_1_x = torch.tensor(set_1_x, dtype=torch.float32)
    set_1_y = torch.tensor(set_1_y, dtype=torch.float32).unsqueeze(1)
    # Combine the individual tensors into training set, validation set, and test set
    # Referenced: https://datascience.stackexchange.com/questions/45916/loading-own-train-data-and-labels-in-dataloader-using-pytorch
    training_set_for_nn = TensorDataset(set_1_x, set_1_y)
    # Initialize dataloaders 
    training_set_dataloader = DataLoader(dataset=training_set_for_nn, batch_size=10, shuffle=True)
    
    # Load in models 
    nn_model = joblib.load("models/saved_models/neural_network.sav")
    log_reg_model = joblib.load("models/saved_models/logistic_regression.sav")
    models = [("nn", nn_model), ("log_reg", log_reg_model)]
    
    # Referenced: https://machinelearningmastery.com/weighted-average-ensemble-with-python/
    # Scores of each model based on accuracy using each of the training set
    
    
    
    scores = obtain_scores_for_base_models(models, train_x, train_y, validation_x, validation_y, training_set_dataloader, validation_set_dataloader)
    print(scores)
    
    
    # Create ensemble and compute the average prediction 

    
