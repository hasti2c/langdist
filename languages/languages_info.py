import pandas as pd
from matplotlib import pyplot as plt

## Language List ##
import lang2vec.lang2vec as l2v
LANGUAGES = pd.DataFrame(index=sorted(list(l2v.DISTANCE_LANGUAGES)))
LANGUAGES.index.name = "code"
LANGUAGES["URIEL"] = LANGUAGES.index.isin(l2v.URIEL_LANGUAGES)
LANGUAGES["learned"] = LANGUAGES.index.isin(l2v.LEARNED_LANGUAGES)
LANGUAGES["distance"] = LANGUAGES.index.isin(l2v.DISTANCE_LANGUAGES)
LANGUAGES.to_csv("languages/languages.csv")

## Venn ##
from matplotlib_venn import venn2, venn3
def langs_venn2(t1, t2):
  c1, c2 = LANGUAGES[t1], LANGUAGES[t2]
  # Ab, aB, AB
  venn2(subsets=((c1 & ~c2).sum(), (~c1 & c2).sum(), (c1 & c2).sum()), set_labels=(t1, t2))
  plt.savefig(f"languages/venn/venn2_{t1}_{t2}.png")
  plt.close()

def langs_venn3(t1, t2, t3):
  c1, c2, c3 = LANGUAGES[t1], LANGUAGES[t2], LANGUAGES[t3]
  # Ab, aB, AB
  venn3(subsets=((c1 & ~c2 & ~c3).sum(), (~c1 & c2 & ~c3).sum(), (c1 & c2 & ~c3).sum(), (~c1 & ~c2 & c3).sum(),
                   (c1 & ~c2 & c3).sum(), (~c1 & c2 & c3).sum(), (c1 & c2 & c3).sum()), set_labels=(t1, t2, t3))
  plt.savefig(f"languages/venn/venn3_{t1}_{t2}_{t3}.png")
  plt.close()

langs_venn2("URIEL", "learned")
langs_venn2("URIEL", "distance")
langs_venn2("learned", "distance")
langs_venn3("URIEL", "learned", "distance")
langs_venn3("URIEL", "distance", "learned")
langs_venn3("learned", "URIEL", "distance")

## ISO-639 ##
from iso639 import languages as iso_languages
LANGUAGES["name"] = "-"
for code in LANGUAGES.index:
  if LANGUAGES.loc[code, "name"] == "-":
    try:
      LANGUAGES.loc[code, "name"] = iso_languages.get(part3=code).name
    except KeyError:
      LANGUAGES.loc[code, "name"] = "?"
LANGUAGES.to_csv("languages/languages.csv")

## Taxonomy ##
import csv
tax = pd.read_csv("languages/lang2tax_code_manual.csv", quoting=csv.QUOTE_NONE)
tax.astype({"joshi": int})
tax.loc[tax.loc[tax["name"] == "min nan"].index, "code"] = "nan"
tax.set_index("code", inplace=True)
tax.drop(index=tax.loc[pd.isna(tax.index)].index, inplace=True)
# TODO The following are repeated in the Joshi file.
# tax.drop(index=tax.loc[tax["name"].isin(["ndonga", "choctaw"]) & (tax["joshi"] == 0)].index, inplace=True)
tax.drop(index=tax.loc[(tax.index == "grn") & (tax["joshi"] == 0)].index, inplace=True)

LANGUAGES = LANGUAGES.merge(tax, 'left', left_index=True, right_index=True)
LANGUAGES.drop(columns=["name_y"], inplace=True)
LANGUAGES.rename(columns={"name_x": "name"}, inplace=True)
LANGUAGES.to_csv("languages/languages.csv")

## Glottolog ##
from pyglottolog import Glottolog
from pyglottolog.languoids import Languoid
glottolog = Glottolog('glottolog-4.8')
languoids = glottolog.languoids_by_code()
LANGUAGES["family"] = "-"
for i, code in enumerate(LANGUAGES.index):
  if LANGUAGES.loc[code, "family"] == "-":
    try:
      LANGUAGES.loc[code, "family"] = languoids[code].family.name
    except KeyError:
      LANGUAGES.loc[code, "family"] = "?"
    except AttributeError:
      LANGUAGES.loc[code, "family"] = "?"
  if (i + 1) % 100 == 0:
    LANGUAGES.to_csv("languages/languages.csv")
LANGUAGES.to_csv("languages/languages.csv")
