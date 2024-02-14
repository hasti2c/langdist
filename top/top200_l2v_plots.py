import numpy as np
import pandas as pd

## Loading Data ##
import lang2vec.lang2vec as l2v

L2V_DATA = {}
for dtype in l2v.DISTANCES:
  L2V_DATA[dtype] = pd.read_csv(f"top/l2v/{dtype}/{dtype}_data.csv", index_col="code", keep_default_na=False)
  # L2V_DATA[dtype].replace("", np.nan, inplace=True)
  # L2V_DATA[dtype] = L2V_DATA[dtype].astype({"joshi": "Float64"}).astype({"joshi": "Int64"})

## Plotting ##
from matplotlib import pyplot as plt
import seaborn as sns

for dtype in l2v.DISTANCES:
  l2v_data = L2V_DATA[dtype]
  if "NA properties" in l2v_data:
    sns.histplot(l2v_data["NA properties"], bins=20)
    plt.savefig(f"top/l2v/{dtype}/{dtype}_NA_properties.png")
    plt.clf()

    g = sns.histplot(l2v_data, x="NA properties", hue="family", multiple="stack", bins=20)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"top/l2v/{dtype}/{dtype}_NA_properties_by_family.png")
    plt.clf()

    g = sns.histplot(l2v_data[l2v_data['joshi'] != ''], x="NA properties", hue="joshi", hue_order=["0.0", "1.0", "2.0", "3.0", "4.0", "5.0"], multiple="stack", bins=20)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"top/l2v/{dtype}/{dtype}_NA_properties_by_joshi.png")
    plt.clf()
