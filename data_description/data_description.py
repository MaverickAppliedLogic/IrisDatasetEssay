import pandas as pd
from sklearn.utils import Bunch


def data_description(data: Bunch):
    df = pd.DataFrame(data.data, columns=data.feature_names)
    print("\nValues: ")
    print("____________________________________")
    print(df)
    df2 = pd.DataFrame()
    df2["species"] = data.target
    df2["species_name"] = df2["species"].map({
        0: data.target_names[0],
        1: data.target_names[1],
        2: data.target_names[2]
    })
    df_species = df2.groupby(["species_name"]).mean()
    df_species["count"] = df2.value_counts("species_name")
    print("\nLabels: ")
    print("____________________________________")
    print(df_species)