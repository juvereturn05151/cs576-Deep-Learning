"""
File Name:    deep_learning_runner.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""
import os
import csv
import numpy as np
import math_utility
import model
import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class DeepLearningRunner:
    def __init__(self):
        self.initialize_data()
        self.model = model.Model([
            layers.Linear(2, 2, seed=1),
            layers.Linear(2, 2, seed=2),
            layers.Linear(2, 1, seed=3),
        ])
        self.model_name = " Linear"

    def replace_model(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def initialize_data(self):
        print(os.listdir())

        X, T = self.load_dataset("multiply_dataset.csv")

        print("Inputs shape:", X.shape)
        print("Targets shape:", T.shape)

        self.X_train, self.X_test, self.T_train, self.T_test = train_test_split(X, T, test_size=0.2, random_state=42)

    def load_dataset(self, csv_path):
        X = []
        T = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            # skip header
            next(reader)

            for row in reader:
                x = float(row[0])
                y = float(row[1])
                out = float(row[2])

                # input: (x, y)
                X.append([x, y])
                # output: (x*y)
                T.append([out])

        return np.array(X), np.array(T)

    def train(self, epochs=500, lr=0.1, batch_size=8):
        N = self.X_train.shape[0]
        rng = np.random.default_rng(42)

        # history for visualization
        self.train_rmse_history = []
        self.test_rmse_history = []
        self.epoch_history = []

        for epoch in range(1, epochs + 1):
            # shuffle each epoch
            idx = rng.permutation(N)
            Xs = self.X_train[idx]
            Ts = self.T_train[idx]

            for i in range(0, N, batch_size):
                xb = Xs[i:i + batch_size]
                tb = Ts[i:i + batch_size]

                # forward
                yb = self.model.forward(xb)

                # loss gradient
                gradient_error = math_utility.mse_backward(yb, tb)

                # backward
                self.model.backward(gradient_error)

                # update
                self.model.update(lr)

            # epoch metrics
            train_pred = self.model.forward(self.X_train)
            test_pred  = self.model.forward(self.X_test)

            train_rmse = math_utility.rmse(train_pred, self.T_train)
            test_rmse  = math_utility.rmse(test_pred, self.T_test)

            self.epoch_history.append(epoch)
            self.train_rmse_history.append(train_rmse)
            self.test_rmse_history.append(test_rmse)

            if epoch == 1 or epoch % 50 == 0:
                print(f"Epoch {epoch:4d} | Train RMSE: {train_rmse:.6f} | Test RMSE: {test_rmse:.6f}")

        self.plot_rmse()

    def plot_rmse(self):
        if not hasattr(self, "epoch_history") or len(self.epoch_history) == 0:
            print("No RMSE history to plot. Train first.")
            return

        plt.figure()
        plt.plot(self.epoch_history, self.train_rmse_history, label="Train RMSE")
        plt.plot(self.epoch_history, self.test_rmse_history, label="Test RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("RMSE over Training Epochs" + self.model_name)
        plt.legend()
        plt.grid(True)

        plt.show()

    def run(self):
        print("Running Deep Learning Runner")
        self.train(epochs=500, lr=0.05, batch_size=8)

        # quick sanity tests
        X_test_points = np.array([[0.2, 0.7], [0.5, 0.5], [1.0, 0.3]], dtype=np.float64)
        preds = self.model.forward(X_test_points).reshape(-1)

        print("\nQuick tests:")
        for (x, y), p in zip(X_test_points, preds):
            print(f"x={x:.2f}, y={y:.2f} -> pred={p:.6f}, true={x*y:.6f}")