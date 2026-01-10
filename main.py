
from sklearn.datasets import load_iris

from data_description.data_description import data_description
from models.SVM.model_radial_svm import ModelRadialSVM
from models.K_means.model_k_means import ModelKMeans
from stadistic_analysis.stadistic_analysis import stadistic_analysis

if __name__ == "__main__":

    iris = load_iris()
    data_description(iris)
    stadistic_analysis(iris)
    rsvm = ModelRadialSVM(iris)
    #rsvm.exec(show_graph=True)
    mkm = ModelKMeans(iris)
    mkm.exec(show_graph=True)









