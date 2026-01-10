import numpy as np
from matplotlib.colors import Normalize, ListedColormap
from scipy.cluster.hierarchy import centroid
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
from sklearn.utils import Bunch
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mlxtend


class ModelKMeans:

 iris: Bunch
 x = None
 model = None

 def __init__(self, iris: Bunch):
     self.iris = iris
     self.generate_model()


 def generate_model(self):
        data = pd.DataFrame(data=self.iris.data, columns=self.iris.feature_names)
        self.x = data[["petal length (cm)", "petal width (cm)"]].values
        self.model = KMeans(n_clusters=3, random_state=42, verbose=0)


 def exec(self, show_graph: bool):
     self.model.fit(self.x)
     labels = self.model.labels_
     print(labels.flatten())
     print(self.iris.target.flatten())
     cm = pd.crosstab(
         self.iris.target,
         labels,
         rownames=['Real'],
         colnames=['Prediction']
     )
     row_ind, col_ind = linear_sum_assignment(-cm)
     mapper = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
     mapper_vector = np.vectorize(mapper.get)
     mapped_labels = mapper_vector(labels)

     accuracy = accuracy_score(
         y_true=self.iris.target,
         y_pred=mapped_labels,
         normalize=True
     )
     print()
     print(f"Precisión del test: {100 * accuracy}%")

     cm = pd.crosstab(
         self.iris.target,
         mapped_labels,
         rownames=['Real'],
         colnames=['Prediction']
     )
     print("Matriz de confusion ")
     print()
     print(cm)

     if show_graph:
         self.show(mapped_labels)



 def show(self, labels):
     fig, ax = plt.subplots()
     fig1, ax1 = plt.subplots()

     centroids = self.model.cluster_centers_
     ax.scatter(self.x[:,0], self.x[:,1],
                c=labels, cmap='Set1', s=60, marker="X")
     ax.scatter(centroids[:,0], centroids[:,1], c='blue', s=60, marker='o', edgecolors='black')
     ax.set_xlabel("Petal length (cm)")
     ax.set_ylabel("Petal width (cm)")
     ax.set_title("K-Means clustering")

     ax1.scatter(self.x[:,0], self.x[:,1],
                 c=self.iris.target, cmap='Set1', s=60, marker="X")
     ax1.set_xlabel("Petal length (cm)")
     ax1.set_ylabel("Petal width (cm)")
     ax1.set_title("Real clustering")
     plt.show()

