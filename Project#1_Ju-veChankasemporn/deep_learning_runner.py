"""
File Name:    deep_learning_runner.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""
import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split

def getRMSE(self, y_pred):
    return np.sqrt(np.mean((self.y_test - y_pred) ** 2))

class DeepLearningRunner:
    def __init__(self):
        self.initialize_data()
    
    def initialize_data(self):
        print(os.listdir())

        X, T = self.load_dataset("multiply_dataset.csv")

        print("Inputs shape:", X.shape)  # (N, 2)
        print("Targets shape:", T.shape)  # (N, 1)

        self.X_train, self.X_test, self.T_train, self.T_test = train_test_split(X, T, test_size=0.2, random_state=42)

    def load_dataset(self, csv_path):
        X = []
        T = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header

            for row in reader:
                x = float(row[0])
                y = float(row[1])
                out = float(row[2])

                X.append([x, y])  # input: (x, y)
                T.append([out])  # output: (x*y)

        return np.array(X), np.array(T)

    def run(self):
        print("Running Deep Learning Runner")