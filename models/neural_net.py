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

def vectorize(the_dataset): 
    """This function vectorize the data"""
    # https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    # Vectorize 
    ct = ColumnTransformer(
        [("division", TfidfVectorizer(stop_words = 'english'), "division"),
         ("dept", TfidfVectorizer(stop_words = 'english'), "dept"),
         ("course", TfidfVectorizer(stop_words = 'english'), "course"),
         ("term", TfidfVectorizer(stop_words = 'english'), "term"),
         ("year", MinMaxScaler(feature_range = (0, 1)), "year"),
         ("item 1", MinMaxScaler(feature_range = (0, 1)), "item 1 (i found the course intellectually stimulating)"),
         ("item 2", MinMaxScaler(feature_range = (0, 1)), "item 2 (the course provided me with a deep understanding of the subject manner)"), 
         ("item 3", MinMaxScaler(feature_range = (0, 1)), "item 3 (the instructor created a course atmosphere that was condusive to my learning)"), 
         ("item 4", MinMaxScaler(feature_range = (0, 1)), "item 4 (course projects, assignments, tests, and/or exams improved my understanding of the course material)"),
         ("item 5", MinMaxScaler(feature_range = (0, 1)), "item 5 (course projects, assignments, tests, and/or exams provided opportunity for me to demonstrate an understanding of the course material)"), 
         ("item 6", MinMaxScaler(feature_range = (0, 1)), "item 6 (overall, the quality of my learning experience in the course was:)"),
         ("instructor generated enthusiasm", MinMaxScaler(feature_range = (0, 1)), "instructor generated enthusiasm"),
         ("course workload", MinMaxScaler(feature_range = (0, 1)), "course workload"),
         ("i would recommend this course", MinMaxScaler(feature_range = (0, 1)), "i would recommend this course"),
         ]
    )
    
    # print(vectorized_data)
    
def split_dataset():
    """This function splits the dataset into training set, validation set, and test set"""
    
if __name__ == '__main__':
    # Read in CSV file 
    the_dataset = pd.read_csv("../data/clean_data/new_data.csv")
    # Vectorize dataset
    vectorized_dataset = vectorize(the_dataset)