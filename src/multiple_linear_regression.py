import numpy as np

class MultipleLinearRegressor:
    def __init__(self, default_intercept=0, default_slope=0) -> None:
        self._intercept = default_intercept  # Now a protected attribute
        self._slope = default_slope          # Now a protected attribute

    def train(self, x: np.array, y: np.array) -> None:
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("x and y must be numpy arrays!")

        xIntercept = np.hstack([np.ones((x.shape[0], 1)), x])

        try:
            inverted = np.linalg.inv(np.dot(xIntercept.T, xIntercept))
        except np.linalg.LinAlgError:
            raise Exception("the matrix X^T X is singular and non-invertible!")

        weights = np.dot(inverted, np.dot(xIntercept.T, y))
        self._intercept = weights[0]  # Update to protected attribute
        self._slope = weights[1:]     # Update to protected attribute

    def predict(self, x: np.array) -> np.array:
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy array!")

        xIntercept = np.hstack([np.ones((x.shape[0], 1)), x])
        return np.dot(xIntercept, np.hstack([self._intercept, self._slope]))

    def get_params(self):
        return {
            'intercept': self._intercept,
            'slope': self._slope.tolist() if hasattr(self._slope, "tolist") else self._slope
        }

    def set_params(self, params: dict) -> None:
        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary!")

        self._intercept = params['intercept']
        self._slope = params['slope']

    # Uncomment below to use the plotting functionality

    '''
    # plotting the results of sklearn and of our own to make visual comparison using matplotlib

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
