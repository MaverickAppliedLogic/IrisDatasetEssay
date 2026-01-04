from matplotlib.pyplot import title
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_description.data_description import data_description
from stadistic_analysis.stadistic_analysis import stadistic_analysis

if __name__ == "__main__":

    iris = load_iris()
    data_description(iris)
    stadistic_analysis(iris)







