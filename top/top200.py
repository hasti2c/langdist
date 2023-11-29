import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

LANGUAGES = pd.read_csv("languages/languages.csv", index_col="code")
TOP = pd.read_csv("top/top200.csv", index_col="code")
LANGUAGES = LANGUAGES.loc[LANGUAGES.index.isin(TOP.index)]
LANGUAGES.astype({"joshi": "Int64"})
# vjk is in top200 but not l2v.
TOP.merge(LANGUAGES, "left", "code").to_csv("top/top200_data.csv")

## Venn ##
from matplotlib_venn import venn2, venn3
def langs_venn2(t1, t2):
  c1, c2 = LANGUAGES[t1], LANGUAGES[t2]
  # Ab, aB, AB
  venn2(subsets=((c1 & ~c2).sum(), (~c1 & c2).sum(), (c1 & c2).sum()), set_labels=(t1, t2))
  plt.savefig(f"top/venn/venn2_{t1}_{t2}.png")
  plt.clf()

def langs_venn3(t1, t2, t3):
  c1, c2, c3 = LANGUAGES[t1], LANGUAGES[t2], LANGUAGES[t3]
  # Ab, aB, AB
  venn3(subsets=((c1 & ~c2 & ~c3).sum(), (~c1 & c2 & ~c3).sum(), (c1 & c2 & ~c3).sum(), (~c1 & ~c2 & c3).sum(),
                   (c1 & ~c2 & c3).sum(), (~c1 & c2 & c3).sum(), (c1 & c2 & c3).sum()), set_labels=(t1, t2, t3))
  plt.savefig(f"top/venn/venn3_{t1}_{t2}_{t3}.png")
  plt.clf()

langs_venn2("URIEL", "learned")
langs_venn2("URIEL", "distance")
langs_venn2("learned", "distance")
langs_venn3("URIEL", "learned", "distance")
langs_venn3("URIEL", "distance", "learned")
langs_venn3("learned", "URIEL", "distance")

## Count Plots
import seaborn as sns
sns.set_palette('tab20')

sns.countplot(LANGUAGES, x="joshi")
plt.tight_layout()
plt.savefig("top/countplot/joshi.png")
plt.close()

sns.countplot(LANGUAGES, x="family")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("top/countplot/family.png")
plt.close()
plt.xticks(rotation=0)

fams = list(LANGUAGES.groupby("family", dropna=False).size().sort_values(ascending=False).index)
index = pd.MultiIndex.from_product([LANGUAGES["family"].unique(), LANGUAGES["joshi"].unique()], names=["family", "joshi"])
count = LANGUAGES.groupby(["family", "joshi"], dropna=False).size().reindex(index, fill_value=0).reset_index()
count = count.pivot(columns="family", index="joshi", values=0)[fams]

g = count.plot(kind='bar', stacked=True)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("top/countplot/joshi_family.png")
plt.close()

g = count.drop(index=np.nan).plot(kind='bar', stacked=True)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("top/countplot/joshi_family_dropna.png")
plt.close()

g = count.T.plot(kind='bar', stacked=True)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("top/countplot/family_joshi.png")
plt.close()
plt.xticks(rotation=0)

g = count.T.drop(columns=np.nan).plot(kind='bar', stacked=True)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("top/countplot/family_joshi_dropna.png")
plt.close()
plt.xticks(rotation=0)