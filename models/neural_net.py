import pandas as pd 
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

"""
Script implementing neural network 
"""
# Referenced: https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# Referenced: https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/
# Referenced: https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch

def vectorize_features(the_dataset): 
    """This function vectorize the features in the dataset"""
    # https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    # Vectorize 
    ct_features = ColumnTransformer(
        [("division", TfidfVectorizer(), "division"),
         ("dept", TfidfVectorizer(), "dept"),
         ("course", TfidfVectorizer(), "course"),
         ("term", TfidfVectorizer(), "term"),
         ("year", MinMaxScaler(feature_range = (0, 1)), ["year"]),
         ("item 1", MinMaxScaler(feature_range = (0, 1)), ["item 1 (i found the course intellectually stimulating)"]),
         ("item 2", MinMaxScaler(feature_range = (0, 1)), ["item 2 (the course provided me with a deep understanding of the subject manner)"]), 
         ("item 3", MinMaxScaler(feature_range = (0, 1)), ["item 3 (the instructor created a course atmosphere that was condusive to my learning)"]), 
         ("item 4", MinMaxScaler(feature_range = (0, 1)), ["item 4 (course projects, assignments, tests, and/or exams improved my understanding of the course material)"]),
         ("item 5", MinMaxScaler(feature_range = (0, 1)), ["item 5 (course projects, assignments, tests, and/or exams provided opportunity for me to demonstrate an understanding of the course material)"]), 
         ("item 6", MinMaxScaler(feature_range = (0, 1)), ["item 6 (overall, the quality of my learning experience in the course was:)"]),
         ("instructor generated enthusiasm", MinMaxScaler(feature_range = (0, 1)), ["instructor generated enthusiasm"]),
         ("course workload", MinMaxScaler(feature_range = (0, 1)), ["course workload"]),
         ("i would recommend this course", MinMaxScaler(feature_range = (0, 1)), ["i would recommend this course"]),
         ("last name", TfidfVectorizer(), "last name"),
         ("description", TfidfVectorizer(stop_words = 'english'), "description"
          )]
    )
    vectorized_data = ct_features.fit_transform(the_dataset)
    # print(vectorized_data)
    return vectorized_data
    
def split_dataset(features, targets):
    """This function splits the dataset into training set, validation set, and test set"""
    # Split the entire dataset RANDOMLY into 70% training, 15% validation, and 15% test examples 
    train_x, test_validation_x, train_y, test_validation_y = train_test_split(features.toarray(), 
                                                                              targets, train_size = 0.7,
                                                                              test_size=0.3, random_state=100)
    validation_x, test_x, validation_y, test_y = train_test_split(test_validation_x, test_validation_y, 
                                                                  train_size = 0.5, test_size=0.5, 
                                                                  random_state= 100)
    return [train_x, train_y, validation_x, validation_y, test_x, test_y]
    
if __name__ == '__main__':
    # Read in CSV file 
    the_dataset = pd.read_csv("data/clean_data/new_data.csv")
    # Vectorize the features in the dataset
    vectorized_dataset_features = vectorize_features(the_dataset)
    print(vectorized_dataset_features.shape)
    # Dataset containing the targets
    target_dataset = the_dataset["recommended"]
    print(target_dataset.shape)
    # Split dataset into training set, validation set, and test set
    train_x, train_y, validation_x, validation_y, test_x, test_y = split_dataset(vectorized_dataset_features, target_dataset)
    # print(train_x.shape)
    # print(train_y.shape)
    
    # print(validation_x.shape)
    # print(validation_y.shape)
    
    # print(test_x.shape)
    # print(test_y.shape)