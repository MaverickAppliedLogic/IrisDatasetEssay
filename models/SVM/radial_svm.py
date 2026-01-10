from sklearn.utils import Bunch
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV

class RadialSVM:

    iris: Bunch
    x = None
    y = None
    X_train = None
    X_test = None
    Y_train = None
    Y_test = None
    modelo = None
    

    def __init__(self, iris: Bunch):
        self.iris = iris
        self.generate_model()

    def generate_model(self):
        data = pd.DataFrame(data=self.iris.data, columns=self.iris.feature_names)
        self.x = data[["petal length (cm)", "petal width (cm)"]].values
        self.y = self.iris.target
        self.X_train, self.X_test, self.Y_train, self.Y_test = (
            train_test_split(self.x, self.y, train_size=0.6, test_size=0.4, random_state=42, shuffle=True))
        param_grid = {'C': np.logspace(-5, 7, 20)}
        function = svm.SVC(kernel="rbf", gamma="scale")
        grid = GridSearchCV(function, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=3, verbose=0,
                            return_train_score=True)

        __ = grid.fit(self.X_train, self.Y_train)

        self.modelo = grid.best_estimator_
        

    def model_stats(self, graph: bool):
        
        pred = self.modelo.predict(self.X_test)
        accuracy = accuracy_score(
            y_true=self.Y_test,
            y_pred=pred,
            normalize=True
        )
        print()
        print(f"Precisión del test: {100 * accuracy}%")
        confusion_matrix = pd.crosstab(
            self.Y_test.ravel(),
            pred,
            rownames=['Real'],
            colnames=['Prediction']
        )
        print("Matriz de confusion ")
        print()
        print(confusion_matrix)
        if graph:
            self.show()


    def show(self):
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()

        ax.set_title("Training graph")
        plot_decision_regions(
            X=self.X_train,
            y=self.Y_train.flatten(),
            clf=self.modelo,
            ax=ax
        )

        ax1.set_title("Test graph")
        plot_decision_regions(
            X=self.X_test,
            y=self.Y_test.flatten(),
            clf=self.modelo,
            ax=ax1
        )

        plt.show()