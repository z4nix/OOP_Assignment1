import pandas as pd
import numpy as np
from sklearn import datasets


class MultipleLinearRegressor:
    def __init__(self, default_intercept=0, default_slope=0) -> None:
        self.intercept = default_intercept
        self.slope = default_slope

    def train(self, x: np.array, y: np.array) -> None:
        xIntercept = np.hstack(
            [np.ones((x.shape[0], 1)), x])  # adding a column of ones (to account for the intercept term)

        weights = np.dot(np.linalg.inv(np.dot(xIntercept.T, xIntercept)),
                         np.dot(xIntercept.T, y))  # computing the weights (w = (X^T * X)^-1 * X^T * y)
        # above is the the dot product, of the inverse of the dot product, of the transpose of xIntercept and xIntercept and the dot product of the transpose of xIntercept and y

        self.intercept = weights[0]  # setting  intercept to the first  element of the weights array
        self.slope = weights[1:]  # setting slope to the rest of the elements of the weights array


    def predict(self, x) -> np.array:
        xIntercept = np.hstack(
            [np.ones((x.shape[0], 1)), x])  # adding a column of ones (to account for the intercept term)
        return np.dot(xIntercept, np.hstack([self.intercept,
                                             self.slope]))  # returning the dot product of xIntercept and the weights array (the intercept and the slope) formula: y = w0 + w1*x1 + w2*x2 + ... + wn*xn


if __name__ == "__main__":
    model = MultipleLinearRegressor()
    '''
    x = np.array([[2, 2], [2, 3], [3, 3], [4, 3]])
    #independent variable (feature)
    y = np.array([0, 1, 2, 3]) # dependent variable (target)
    x = x.astype(float)
    x += np.random.rand(*x.shape) # add random noise to x
    print(f"SimpleLinerRegressor coefficients -- intercept {model.intercept} -- slope {model.slope}") # print the intercept and slope
    model.train(x, y) # train the model
    y_pred = model.predict(x) # predict the target variable

    print("Ground truth and predicted values:", y, y_pred, sep="\n")
    '''
    # Test the model on the diabetes dataset
    diabetes = datasets.load_diabetes()
    diabetes_x = diabetes.data
    diabetes_y = diabetes.target
    diabetes_x = diabetes_x.astype(float)
    model.train(diabetes_x, diabetes_y)
    diabetes_y_pred = model.predict(diabetes_x)
    print("Ground truth and predicted values:", diabetes_y, diabetes_y_pred, sep="\n")

    # compare with sklearn
    from sklearn.linear_model import LinearRegression

    sklearn_model = LinearRegression()
    sklearn_model.fit(diabetes_x, diabetes_y)
    sklearn_y_pred = sklearn_model.predict(diabetes_x)
    print("Ground truth and SKLEARN predicted values:", diabetes_y, sklearn_y_pred, sep="\n")


'''
    # plotting the results of sklearn and of my own to make visual comparison using matplotlib

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    # plot for custom model
    plt.subplot(1, 2, 1)
    plt.scatter(diabetes_y, diabetes_y_pred, color='blue', label='custom model predictions')
    plt.plot([diabetes_y.min(), diabetes_y.max()], [diabetes_y.min(), diabetes_y.max()], 'k--', lw=2)
    plt.xlabel('measured')
    plt.ylabel('predicted')
    plt.title('custom model')
    plt.legend()

    # Plot for sklearn model
    plt.subplot(1, 2, 2)
    plt.scatter(diabetes_y, sklearn_y_pred, color='green', label='sklearn model predictions')
    plt.plot([diabetes_y.min(), diabetes_y.max()], [diabetes_y.min(), diabetes_y.max()], 'k--', lw=2)
    plt.xlabel('measured')
    plt.ylabel('predicted')
    plt.title('Sklearn Model')
    plt.legend()

    plt.tight_layout()
    plt.show()
'''