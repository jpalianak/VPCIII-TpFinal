import pandas as pd
import matplotlib.pyplot as plt
from .utils import label2id

id2labels = {v: k for k, v in label2id.items()}



def plot_class_distribution(split_name, split):
    df = pd.DataFrame(split["labels"], columns=["label_id"])
    df["label_name"] = df["label_id"].map(id2labels)
    df["label_name"].value_counts().sort_index().plot(kind="bar", title=f"Distribución en {split_name}")
    plt.xticks(rotation=45)
    plt.ylabel("Cantidad")
    plt.show()

    