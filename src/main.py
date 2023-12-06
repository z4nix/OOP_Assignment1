from multiple_linear_regression import MultipleLinearRegressor
from sklearn import datasets
from regression_plotter import RegressionPlotter
from model_saver import ModelSaver

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


    # plot the regression lines for each feature
    plotter = RegressionPlotter(model, diabetes.data, diabetes.target)
    plotter.plot()

    # save and load parameters
    saver = ModelSaver(format='json')
    saver.save_model(model, 'model_params.json')

    saver.load_model('model_params.json', model)