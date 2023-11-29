import pandas as pd
import csv

LANGUAGES = pd.read_csv("languages/languages.csv", header=None, quoting=csv.QUOTE_NONE)
tax = pd.read_csv("languages/lang2tax.csv", header=None, quoting=csv.QUOTE_NONE)
tax.columns = ["name", "joshi"]
tax.astype({"joshi": int})
# TODO The following are repeated in the Joshi file.
tax.drop(index=tax.loc[tax["name"].isin(["ndonga", "choctaw"]) & (tax["joshi"] == 0)].index, inplace=True)

LANGUAGES["name lower"] = LANGUAGES["name"].apply(str.lower)
LANGUAGES["code copy"] = LANGUAGES.index
left = LANGUAGES.merge(tax, 'left', left_on="name lower", right_on="name")
right = LANGUAGES.merge(tax, 'right', left_on="name lower", right_on="name")

left.set_index("code copy", inplace=True)
left.index.name = "code"
left.drop(columns=["name lower", "name_y"], inplace=True)
left.rename(columns={"name_x": "name"}, inplace=True)
LANGUAGES = left
LANGUAGES.to_csv("languages/languages.csv")

right.set_index("name_y", inplace=True)
right.index.name = "name"
right.drop(columns=["URIEL", "learned", "distance", "name_x", "name lower"], inplace=True)
right.rename(columns={"code copy": "code"}, inplace=True)
right.to_csv("languages/lang2tax_code.csv")
