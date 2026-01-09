import numpy as np
from scipy.stats import alpha
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV

from data_description.data_description import data_description
from stadistic_analysis.stadistic_analysis import stadistic_analysis

if __name__ == "__main__":

    iris = load_iris()
    data_description(iris)
    stadistic_analysis(iris)
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    X = data[["petal length (cm)", "petal width (cm)"]].values
    y = iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
    param_grid = {'C' : np.logspace(-5, 7, 20)}
    function = svm.SVC(kernel="rbf", gamma="scale")
    grid = GridSearchCV(function, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=3, verbose=0,
                        return_train_score= True)

    __ = grid.fit(X_train, Y_train)
    results = pd.DataFrame(grid.cv_results_)
    results.filter(regex='(params.*|mean_|std_t)')
    results.drop(columns='params')
    results.sort_values('mean_test_score', ascending=False)
    results.head(5)

    print("________________________________________")
    print("Best Params:")
    print("_________________________________________")
    print(grid.best_params_, grid.best_score_, grid.scoring)

    modelo = grid.best_estimator_

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))

    grid = np.vstack([xx.ravel(), yy.ravel()]).T

    pred_grid = modelo.predict(grid)

    fig, ax = plt.subplots()
    ax.scatter(grid[:,0], grid[:,1], c=pred_grid, alpha= 0.05)
    ax.scatter(X_train[:,0],X_train[:,1], c=Y_train, alpha=1, edgecolors='black')


    Z = pred_grid.reshape(xx.shape)

    ax.contour(xx, yy, Z,colors='k', levels=[0,1,2],alpha=1,linestyles=['dotted', 'dotted'])


    plt.show()







