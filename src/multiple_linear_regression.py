import numpy as np
from sklearn import datasets
from regression_plotter import RegressionPlotter
from model_saver import ModelSaver


class MultipleLinearRegressor:
    def __init__(self, default_intercept=0, default_slope=0) -> None:
        self.intercept = default_intercept
        self.slope = default_slope

    def train(self, x: np.array, y: np.array) -> None:
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("x and y must be numpy arrays!")

        xIntercept = np.hstack(
            [np.ones((x.shape[0], 1)), x])  # adding a column of ones (to account for the intercept term)

        try:
            inverted = np.linalg.inv(np.dot(xIntercept.T, xIntercept)) #checking if the matrix is invertible
        except np.linalg.LinAlgError:
            raise Exception("the matrix X^T X is singular and non-invertible!")

        weights = np.dot(inverted, np.dot(xIntercept.T, y))  # computing the weights (w = (X^T * X)^-1 * X^T * y)
        # above is the the dot product, of the inverse of the dot product, of the transpose of xIntercept and xIntercept and the dot product of the transpose of xIntercept and y
        self.intercept = weights[0]  # setting  intercept to the first  element of the weights array
        self.slope = weights[1:]  # setting slope to the rest of the elements of the weights array

    def predict(self, x: np.array) -> np.array:
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy array!")

        xIntercept = np.hstack([np.ones((x.shape[0], 1)), x])  # adding a column of ones (to account for the intercept term)
        return np.dot(xIntercept, np.hstack([self.intercept,
                                             self.slope]))  # returning the dot product of xIntercept and the weights array (the intercept and the slope) formula: y = w0 + w1*x1 + w2*x2 + ... + wn*xn

    def get_params(self):
        return {
            'intercept': self.intercept,
            'slope': self.slope.tolist() if hasattr(self.slope, "tolist") else self.slope  # convert to
        }

    def set_params(self, params: dict) -> None:
        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary!")

        self.intercept = params['intercept']
        self.slope = params['slope']


if __name__ == "__main__":
    model = MultipleLinearRegressor()

    # test the model on the diabetes dataset
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
    # plot the regression lines for each feature
    plotter = RegressionPlotter(model, diabetes.data, diabetes.target)
    plotter.plot()
    '''

    '''
    # save and load parameters
    saver = ModelSaver(format='json')
    saver.save_model(model, 'model_params.json')

    saver.load_model('model_params.json', model)
    '''

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
