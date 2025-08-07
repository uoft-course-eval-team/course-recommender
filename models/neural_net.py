import pandas as pd 
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import joblib
import shap

"""
Script implementing a feed-forward neural network model that uses the Adam optimizer.
"""
# Global variables 
batch_size = 10 # Batch size
learning_rate = 0.0001 # Learning rate
epochs_num = 100 # Epochs
hidden_size = 50 # Default hidden layer dimension
output_size = 1
reg_penalty = 0.001 # Regularization penalty (Prevents overfitting)
max_features_num = 80000 # To reduce the number of dimensions!

# torch.autograd.set_detect_anomaly(True) # Stops the script if it encounters any errors

class NeuralNetwork(nn.Module):
    """A neural network that inherits from PyTorch's neural network"""
    # Referenced: https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch
    def __init__(self, input_num, hidden_num, output_num):
        super().__init__()
        self.first_layer = nn.Linear(input_num, hidden_num)
        self.last_layer = nn.Linear(hidden_num, output_num)

    def forward(self, x):
        """
        Applies activation functions to the different layers to get the final output from 
        performing forward pass 
        """
        out = torch.nn.functional.relu(self.first_layer(x)) 
        out = torch.nn.functional.sigmoid(self.last_layer(out))
        return out

def vectorize_features(the_dataset): 
    """This function vectorize the features in <the_dataset>"""
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
         ("description", TfidfVectorizer(stop_words = 'english', max_features = max_features_num), "description"
          )],
        sparse_threshold=0.0 # Parameter suggested by ChatGPT to fix NaN issue during training
    )
    vectorized_data = ct_features.fit_transform(the_dataset)
    # Convert all nan to numbers
    # https://numpy.org/doc/2.1/reference/generated/numpy.nan_to_num.html
    vectorized_data = np.nan_to_num(vectorized_data.astype(np.float32), nan = 0.0, posinf = 1.0, neginf = 0) # Function suggested by ChatGPT to fix NaN issue during training
    return vectorized_data, ct_features
    
def split_dataset(features, targets):
    """This function splits the dataset into training set, validation set, and test set where <features> represent the 
    data and <targets> represent the class the data represent.
    """
    # Split the entire dataset RANDOMLY into 80% training, 10% validation, and 10% test examples 
    train_x, test_validation_x, train_y, test_validation_y = train_test_split(features, targets, train_size = 0.8,
                                                                              test_size=0.2, random_state=100)
    validation_x, test_x, validation_y, test_y = train_test_split(test_validation_x, test_validation_y, train_size = 0.5, test_size=0.5, 
                                                                  random_state= 100)
    return [train_x, train_y, validation_x, validation_y, test_x, test_y]

def convert_to_dataloaders(train_x, train_y, validation_x, validation_y, test_x, test_y):
    """This function takes in the training sets (<train_x> and <train_y>), validation set 
    (<validation_x> and <validation_y>), and test sets (<test_x> and <test_y>)
    and convert them into tensors then into dataloader objects."""

    # Transform datasets into tensors
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1)
    validation_x = torch.tensor(validation_x, dtype=torch.float32)
    validation_y = torch.tensor(validation_y, dtype=torch.float32).unsqueeze(1)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32).unsqueeze(1)
    
    # Combine the individual tensors into training set, validation set, and test set
    # Referenced: https://datascience.stackexchange.com/questions/45916/loading-own-train-data-and-labels-in-dataloader-using-pytorch
    training_set = TensorDataset(train_x, train_y)
    validation_set = TensorDataset(validation_x, validation_y)
    test_set = TensorDataset(test_x, test_y)

    # Initialize dataloaders 
    training_set_dataloader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
    validation_set_dataloader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True)
    test_set_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    
    # for x, y in training_set_dataloader:
    #     print(f"X's shape: {x.shape}")
    #     print(f"Y's shape: {y.shape}")
    #     break
    # for x, y in validation_set_dataloader:
    #     print(f"X's shape: {x.shape}")
    #     print(f"Y's shape: {y.shape}")
    #     break
    # for x, y in test_set_dataloader:
    #     print(f"X's shape: {x.shape}")
    #     print(f"Y's shape: {y.shape}")
    #     break
    
    return training_set_dataloader, validation_set_dataloader, test_set_dataloader

def training_nn(nn_model, training_set, type_of_set, is_validating):
    """This function trains the neural network <nn_model> using the binary-cross entropy as the loss function
    and Adam to optimize its parameters (weights and biases). It uses <training_set> as the data and the type of set 
    is represented by <type_of_set>. <is_validating> indicates if training_nn is being called by the validation function or not."""
    loss_function = nn.BCELoss()
    objective_function = torch.optim.Adam(nn_model.parameters(), lr = learning_rate, weight_decay = reg_penalty)
    
    # Referenced: https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch
    training_loss_vals = []
    for epoch in range(epochs_num): 
        # Store the total loss per epoch
        total_loss_per_epoch = 0.0
        for data, actual_label in training_set: 
            # Reset gradients
            objective_function.zero_grad()
            # Have model make prediction based on current datapoint 
            predicted_label = nn_model(data)
            # Calculate the loss 
            the_loss = loss_function(predicted_label, actual_label)
            # Compute the gradient
            the_loss.backward()
            objective_function.step()
            # Add the loss to the total loss per epoch count
            total_loss_per_epoch += the_loss.item()
        # Calculate the average loss 
        average_loss_per_epoch = total_loss_per_epoch / len(training_set)
        print(f"Epoch: {epoch}, Average Loss: {average_loss_per_epoch}")
        training_loss_vals += [average_loss_per_epoch]
    
    if is_validating is False: 
        # This means that we are not finding the optimal hidden dimension number with this function and 
        # Not training from ensemble
        # Create visualization 
        plt.figure()
        plt.title(f"Average loss over epochs for {type_of_set} set")
        plt.xlabel("Epochs")
        plt.ylabel("Average Loss")
        plt.plot(range(len(training_loss_vals)), np.array(training_loss_vals))
        plt.savefig(f"graphs/epochs_neural_network_{type_of_set}.png")
    
    return min(training_loss_vals)

def validation(validation_set, k, input_size): 
    """Return the lowest loss for the neural network for the hidden layer dimension, <k>"""
    nn_model = NeuralNetwork(input_size, k, output_size)
    return training_nn(nn_model, validation_set, "validation", True)

def prediction(testing_set, nn_model):
    """Make predictions and return the accuracy value, predicted labels, and actual labels of <testing_set>."""
    # Referenced https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch
    # for loop
    predicted_labels = []
    actual_labels = []
    total = 0
    correct = 0
    with torch.no_grad():
        for datapoint, actual_label in testing_set:
            # Get prediction from model 
            prediction = nn_model(datapoint)  
            # Apply threshold to get predicted labels (1 = is recommended, 0 = somewhat recommended/not recommended)
            predicted_label = np.where(prediction >= 0.5, 1, 0) 
            # Flatten predictions and actual labels
            predicted_label = np.array(list(itertools.chain(*predicted_label))) 
            actual_label = np.array(actual_label).flatten()
            # Append 
            predicted_labels.append(predicted_label)
            actual_labels.append(actual_label)
            # Add to the correct and total variables
            correct += (predicted_label == actual_label).sum()  # Count correct predictions
            total += actual_label.shape[0]
    # Calculate accuracy 
    accuracy_val = correct/total
    return accuracy_val, predicted_labels, actual_labels

def create_report_confusion_matrix(predicted_labels, actual_labels):
    """Prints out classification report and a confusion matrix visualization."""
    # Referenced: https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch
    # Print out classification report
    predicted_labels = list(itertools.chain(*predicted_labels))
    actual_labels = list(itertools.chain(*actual_labels))
    the_report = classification_report(actual_labels, predicted_labels, output_dict=True)
    the_report_df= pd.DataFrame(the_report)
    the_report_df.to_csv("graphs/neural_network_table.csv")
    
    # Create confusion matrix visualizatiuon
    confusion_matrix_results = confusion_matrix(actual_labels, predicted_labels)
    plt.figure()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Feedforward Neural Network Confusion Matrix")
    sns.heatmap(confusion_matrix_results, annot = True, fmt = "g", cbar=False)
    plt.savefig(f"graphs/confusion_matrix_neural_network.png")

def create_shapley_values_graph(nn_model, train_x, train_y, test_x, test_y): 
    """Create graph that provides Shapley values for the purposes of interpretability for the neural network."""
    # Referenced: https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability

    
if __name__ == '__main__':
    # Referenced: https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    # Referenced: https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/
    # Referenced: https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch
    # Referenced: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
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
    # print(train_x.shape)
    # print(train_y.shape)
    # print(validation_x.shape)
    # print(validation_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)
    # Convert the datasets into dataloaders for the neural network
    training_set_dataloader, validation_set_dataloader, test_set_dataloader = convert_to_dataloaders(train_x, train_y, validation_x, validation_y, test_x, test_y)
    input_size = train_x.shape[1]
    
    # Test different values for hidden_size 
    k_values = [16, 30, 50, 60, 65]
    all_loss = []
    lowest_loss = 1.0
    lowest_loss_k = hidden_size 
    for k in k_values: 
        a_loss = validation(validation_set_dataloader, k, input_size)
        all_loss.append(a_loss)
        if lowest_loss > a_loss: 
            lowest_loss = a_loss
    hidden_size = lowest_loss_k
    print(f"Hidden dimension that gives the lowest loss: {hidden_size}")

    # Create graph for validation for hidden_size
    plt.figure()
    plt.title(f"Average loss over different number of nodes in hidden layer")
    plt.xlabel("Number of nodes in hidden layer")
    plt.ylabel("Average Loss")
    plt.plot(k_values, np.array(all_loss))
    plt.savefig(f"graphs/hidden_nodes_neural_network.png")

    # Create and train neural networks on chosen value for hidden_size
    # On validation set (Comment out if you want to run)
    nn_model_validation = NeuralNetwork(input_size, hidden_size, output_size)
    training_nn(nn_model_validation, validation_set_dataloader, "validation", False)
    # On training set
    nn_model = NeuralNetwork(input_size, hidden_size, output_size)
    # Save the trained model 
    joblib.dump(nn_model, "models/saved_models/neural_network.sav")
    training_nn(nn_model, training_set_dataloader, "training", False)
    
    # Make predictions 
    # On training set 
    accuracy_val_training, predicted_labels_training, actual_labels_training = prediction(training_set_dataloader, nn_model)
    # On validation set 
    accuracy_val_validation, predicted_labels_validation, actual_labels_validation = prediction(validation_set_dataloader, nn_model)
    # On test set
    accuracy_val_test, predicted_labels_test, actual_labels_test = prediction(test_set_dataloader, nn_model)
    # Print out the accuracies based on the different sets 
    print(f'Accuracy on training set: {accuracy_val_training}')
    print(f'Accuracy on validation set: {accuracy_val_validation}')
    print(f'Accuracy on test set: {accuracy_val_test}')
    
    # Confusion matrix
    create_report_confusion_matrix(predicted_labels_test, actual_labels_test)
    
    # # Create graph for Shapley values
    # create_shapley_values_graph(loaded_nn_model, train_x, train_y, test_x, test_y)