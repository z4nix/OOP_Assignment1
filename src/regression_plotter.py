import pandas as pd
import numpy as np
from sklearn import datasets
from multiple_linear_regression import MultipleLinearRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class RegressionPlotter:

    def __init__(self, model):
        self.model = model

    def plot2D(self, x, y):
        choice = input("Enter '3d' for 3D plot or '2d' for two 2D plots: ")

        if choice == '3d':
            fig = plt.figure()
            ax = Axes3D(fig)

            #generate a grid of points, to then plot the regression plane
            x0_range = np.linspace(self.x[:, 0].min(), self.x[:, 0].max(), num=100)
            x1_range = np.linspace(self.x[:, 1].min(), self.x[:, 1].max(), num=100)
            x0, x1 = np.meshgrid(x0_range, x1_range)

            #calculate the z-values for each point on the grid using the model and the x0 and x1 vals
            z = self.model.predict(np.array([x0.ravel(), x1.ravel()]).T).reshape(x0.shape)

            #plot original data points and the regression plane on the same plot
            ax.scatter(self.x[:, 0], self.x[:, 1], self.y, color='b')
            ax.plot_surface(x0, x1, z, alpha= 0.5, rstride =100, cstride= 100)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        elif choice == '2d':

            for i in range(2): #plotting the regression line for each feature (2)
                plt.figure()
                plt.scatter(self.x[:, i], self.y, color='b')
                plt.plot(self.x[:, i], self.model.predict(self.x), color='r')

                plt.xlabel(f'Feature {i + 1}')
                plt.ylabel('Target')

        plt.show()

    def plotND(self, x, y) -> None:
        self.model.train(self.x, self.y)
        self.y_pred = self.model.predict(self.x)

        for i in range(self.featureNumber): #plotting the regression line for each feature
            plt.figure()
            plt.scatter(self.x[:, i], self.y, color='black')
            plt.plot(self.x[:, i], self.y_pred, color='blue', linewidth=3)
            plt.xlabel(f'Feature {i + 1}')
            plt.ylabel('Target')
        plt.show()

    def plot(self, x, y):

        self.featureNumber = x.shape[1]

        if self.featureNumber == 2:
            self.plot2D(x,y)
        else:
            self.plotND(x,y)

if __name__ == "__main__":

    plotter = RegressionPlotter(MultipleLinearRegressor())
    plotter.plot(x, y) # x and y are the data to be plotted, they can be defined by the user. For this reason, they have purposely been left empty here