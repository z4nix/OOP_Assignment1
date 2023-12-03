import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class RegressionPlotter:

    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y

    def plot3D(self):
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


    def plotND(self) -> None:
        for i in range(self.featureNumber): 
            plt.figure()
            plt.scatter(self.x[:, i], self.y, color='black')

        # create a grid of values for the current feature
            feature_grid = np.linspace(self.x[:, i].min(), self.x[:, i].max(), 100)

        # create a dataset where all features are their mean values except the current feature
            x_modified = np.full((100, self.featureNumber), self.x.mean(axis=0))
            x_modified[:, i] = feature_grid

        # predict using the modified dataset
            y_pred = self.model.predict(x_modified)

            plt.plot(feature_grid, y_pred, color='blue', linewidth=3, label='Regression Line')
            plt.xlabel(f'Feature {i + 1}')
            plt.ylabel('Target')
            plt.legend()
            plt.show()


    def plot(self):

        # get the number of features
        self.featureNumber = self.x.shape[1]

        if self.featureNumber == 2:
            choice = input("Enter '3d' for 3D plot or '2d' for two 2D plots (default): ")
            if choice == "3d":
                self.plot3D()
            else:
                self.plotND()
        else:
            self.plotND()
