import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
import sklearn.preprocessing

from hebbnet import HebbNetSimple


def generate_blobs(show=False):
    X, y = datasets.make_blobs(centers=2)

    return normalize_data(X), y


def generate_moons(show=False):
    X, y = datasets.make_moons(200, noise=0.0)

    return normalize_data(X), y


def generate_circles():
    X, y = datasets.make_circles(n_samples=500, noise=0.2)

    return normalize_data(X), y


def generate_xor(signed=True):
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    y = np.array([1, 0, 0, 1])

    if signed:
        X = 2 * X - 1
        y = 2 * y - 1

    return X, y


def normalize_data(X):
    # return sklearn.preprocessing.StandardScaler().fit_transform(X)
    return X


def plot_data(X, y, show=True):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    if show:
        plt.show()


def plot_decision_boundary(pred_func, X, y, padding=0.5):
    """
    plot the decision boundary
    :param pred_func: function used to predict the label (numpy -> numpy)
    :param X: input data
    :param y: given labels
    :return:
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.1, cmap=plt.cm.bwr)
    plot_data(X, y, show=False)
    plt.show()


if __name__ == "__main__":
    # Train the linear model
    X, y = generate_blobs()
    model = SGDClassifier()
    model.fit(X, y)

    plot_decision_boundary(model.predict, X, y)
