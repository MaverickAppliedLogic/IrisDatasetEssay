import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import Bunch


def stadistic_analysis(data: Bunch):
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df["species"] = data.target
    estadisticas = df.groupby(["species"]).describe()
    pd.set_option("display.max_columns", None)
    print("Stadistic Basic Values: ")
    print("____________________________________\n")
    print(estadisticas)

    fig, ax = plt.subplots(2, 3)
    sns.histplot(data=df, x=df.index, y="sepal length (cm)", hue="species", ax=ax[0, 0])
    ax[0, 0].set_xlabel("Iris")
    sns.histplot(data=df, x=df.index, y="sepal width (cm)", hue="species", ax=ax[0, 1])
    ax[0, 1].set_xlabel("Iris")
    sns.histplot(data=df, x=df.index, y="petal length (cm)", hue="species", ax=ax[1, 0])
    ax[1, 0].set_xlabel("Iris")
    sns.histplot(data=df, x=df.index, y="petal width (cm)", hue="species", ax=ax[1, 1])
    ax[1, 1].set_xlabel("Iris")
    plt.tight_layout()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax[0, 2])

    df["species"] = df["species"].map({
        0: data.target_names[0],
        1: data.target_names[1],
        2: data.target_names[2]
    })
    sns.pairplot(df, hue="species")

