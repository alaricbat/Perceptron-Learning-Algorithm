# Perceptron Learning Algorithm

A simple Python implementation of the Perceptron Learning Algorithm for binary classification using the banknote authentication dataset.

## Overview

The perceptron is one of the simplest supervised learning algorithms for binary classification. It learns a decision boundary by updating its weights and bias whenever a training example is misclassified. This project implements the classic perceptron update rule and demonstrates it on a real-world classification dataset.

This repository contains:

- A reusable `Perceptron` class implemented in `clazz/Perceptron.py`
- A dataset for banknote authentication in `dataset/data_banknote_authentication.csv`
- A Jupyter notebook showing the full workflow, including data exploration, preprocessing, training, and evaluation

## Project Structure

```text
Perceptron-Learning-Algorithm/
├── clazz/
│   └── Perceptron.py
├── dataset/
│   └── data_banknote_authentication.csv
├── PerceptronLearningAlgorithm.ipynb
├── README.md
└── .DS_Store
```

## Algorithm Details

The perceptron model learns by iterating through training samples and adjusting model parameters when a prediction is wrong:

- Input: feature vector `x`
- Output: binary label `-1` or `1`
- Decision function: `score = x · w + b`
- Prediction: `1` if `score >= 0`, otherwise `-1`
- Update rule for a misclassified sample:
  - `w = w + learning_rate * y * x`
  - `b = b + learning_rate * y`

This implementation supports:

- configurable `epochs`
- configurable `learning_rate`
- fitting on training data
- prediction for new samples

## Dataset

The project uses the Banknote Authentication dataset, which contains 5 numerical features:

- variance
- skewness
- kurtosis
- entropy
- class label (`0` or `1`)

The notebook converts the original class labels to `-1` and `1` for perceptron training, since the perceptron traditionally works with binary labels in the form `{-1, +1}`.

## Installation

Use Python 3 and install the required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

## Usage

### Run the model from Python

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from clazz.Perceptron import Perceptron

# Load data
path = "dataset/data_banknote_authentication.csv"
df = pd.read_csv(path)

# Prepare features and labels
X = df.drop(columns=["class"])
y = df["class"].values

# Standardize features
X_scaled = (X - np.mean(X.values, axis=0)) / np.std(X.values, axis=0)
y = np.where(y == 0, -1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# Train the perceptron
model = Perceptron(learning_rate=0.5, epochs=1000)
model.fit(X_train.to_numpy(), y_train)

# Evaluate
predictions = model.predict(X_test.to_numpy())
print("Accuracy:", accuracy_score(y_test, predictions))
```

### Open the notebook

```bash
jupyter notebook PerceptronLearningAlgorithm.ipynb
```

## Example Results

Using a standard train/test split with the dataset, this implementation achieved approximately 98.18% accuracy on the test set.

```text
Accuracy: 0.9818181818181818
```

Classification report summary:

```text
              precision    recall  f1-score   support

          -1     0.9933    0.9739    0.9835       153
           1     0.9680    0.9918    0.9798       122

    accuracy                         0.9818       275
```

## Notes

- This is a beginner-friendly implementation designed to explain the perceptron learning process.
- The algorithm works best when the dataset is linearly separable or nearly so.
- You can experiment with different values of `learning_rate` and `epochs` to observe how the model changes.

## License

This project is intended for learning and educational use.
