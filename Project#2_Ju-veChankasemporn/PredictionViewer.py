"""
File Name:    PredictionViewer.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

import numpy as np
import matplotlib.pyplot as plt


class PredictionViewer:
    def __init__(self, model):
        self.model = model

    def show_predictions(self, images, labels, num_images=10):
        indices = np.random.choice(len(images), num_images, replace=False)

        plt.figure(figsize=(12, 5))

        for i, idx in enumerate(indices):
            img = images[idx]
            true_label = int(labels[idx])

            # Add batch dimension for prediction
            pred = self.model.predict(np.expand_dims(img, axis=0), verbose=0)
            pred_label = np.argmax(pred)

            plt.subplot(2, int(np.ceil(num_images / 2)), i + 1)
            plt.imshow(img)
            plt.title(
                f"T: {true_label}\nP: {pred_label}",
                color="green" if true_label == pred_label else "red"
            )
            plt.axis("off")

        plt.tight_layout()
        plt.show()