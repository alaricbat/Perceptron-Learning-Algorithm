import numpy as np

class Perceptron():

    def __init__(self, 
                 epochs = 100,
                 learning_rate = 0.05):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.w = None
        self.b = 0.0


    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):

            errors = 0.0

            for i in range(n_samples):

                x_i = X[i]
                y_i = y[i]

                score = (x_i @ self.w) + self.b
                prediction = 1 if score >= 0 else -1

                if prediction != y_i:                 

                    self.w = self.w + self.learning_rate * y_i * x_i

                    self.b += self.learning_rate * y_i

                    errors += 1

            if errors == 0:
                break


        

    def predict(self, X):

        X = np.array(X)

        score = (
            X @ self.w + self.b
        )

        return np.where(score >= 0, 1, -1)