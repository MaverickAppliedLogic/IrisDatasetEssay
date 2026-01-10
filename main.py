
from sklearn.datasets import load_iris

from data_description.data_description import data_description
from models.SVM.radial_svm import RadialSVM
from stadistic_analysis.stadistic_analysis import stadistic_analysis

if __name__ == "__main__":

    iris = load_iris()
    data_description(iris)
    stadistic_analysis(iris)
    rsvm = RadialSVM(iris)
    rsvm.model_stats(graph=True)








