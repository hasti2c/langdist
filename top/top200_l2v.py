import numpy as np
import pandas as pd

## Loading Data ##
import lang2vec.lang2vec as l2v
import scipy.sparse as sparse
from zipfile import ZipFile as zf

LANGUAGES = pd.read_csv("top/top200_data.csv", index_col="code", keep_default_na=False)
LANGUAGES = LANGUAGES.replace({'True': True, 'False': False})
LANGUAGES = LANGUAGES.astype({'URIEL': bool, 'learned': bool, 'distance': bool})
LANGUAGES = LANGUAGES[np.where(LANGUAGES["distance"], True, False)]
all_indices = [l2v.DISTANCE_LANGUAGES.index(lang) for lang in LANGUAGES.index]

DISTANCE_DATA = {}
for dtype in l2v.DISTANCES:
  with zf(l2v.DISTANCES_FILE, 'r') as zp:
    data = sparse.load_npz(zp.open(l2v.map_distance_to_filename(dtype))).todense()
    i_lower, i_upper = np.tril_indices(data.shape[0], -1), np.triu_indices(data.shape[0], 1)
    data[i_lower] = data[i_upper]
    DISTANCE_DATA[dtype] = data[all_indices, :][:, all_indices]

DISTANCE_FEATURE_MAP = {
  "syntactic": "syntax_wals|syntax_sswl|syntax_ethnologue",
  "phonological": "phonology_wals|phonology_ethnologue",
  "inventory": "inventory_ethnologue|inventory_phoible_aa|inventory_phoible_gm|inventory_phoible_saphon|inventory_phoible_spa|inventory_phoible_ph|inventory_phoible_ra|inventory_phoible_upsid"
}
FEATURES_DATA = {}
for dtype in l2v.DISTANCES:
  if dtype in DISTANCE_FEATURE_MAP:
    FEATURES_DATA[dtype] = l2v.get_concatenated_sets(list(LANGUAGES.index), DISTANCE_FEATURE_MAP[dtype])

## Helper Functions ##
def filter(dtype, series):
  indices = [list(LANGUAGES.index).index(lang) for lang in LANGUAGES[series].index]
  return DISTANCE_DATA[dtype][indices, :][:, indices]

## l2v Distance Data ##
L2V_DATA = {}
for dtype in l2v.DISTANCES:
  L2V_DATA[dtype] = LANGUAGES.copy().loc[:, ["URIEL"]]
  data, l2v_data = DISTANCE_DATA[dtype], L2V_DATA[dtype]
  if dtype in FEATURES_DATA:
    feats = FEATURES_DATA[dtype][1]
    l2v_data["NA properties"] = (feats == -1).sum(axis=1)
    l2v_data["non-NA properties"] = (feats != -1).sum(axis=1)
    l2v_data["zero properties"] = (feats == 0).sum(axis=1)
    l2v_data["one properties"] = (feats == 1).sum(axis=1)
  l2v_data.to_csv(f"top/l2v/{dtype}/{dtype}_data.csv")
  # LANG_DISTS[f"mean {dtype} distance"] = np.mean(DATA, axis=1)
  # LANG_DISTS[f"median {dtype} distance"] = np.median(DATA, axis=1)
#   LANG_DISTS[f"zeros in {dtype} distance"] = (DATA == 0).sum(axis=1)
#   LANG_DISTS[f"ones in {dtype} distance"] = (DATA == 1).sum(axis=1)
#   LANG_DISTS[f"rest in {dtype} distance"] = ((DATA != 0) & (DATA != 1)).sum(axis=1)
# LANG_DISTS.to_csv("top/top200_l2v_dists.csv")