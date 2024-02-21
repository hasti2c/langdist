import argparse
import typing as T
import numpy as np
import pandas as pd
import lang2vec.lang2vec.lang2vec as l2v
from scipy import stats
from scipy import sparse
from zipfile import ZipFile as zf


DISTANCE_FEATURE_MAP = {
  "syntactic": "syntax_wals|syntax_sswl|syntax_ethnologue",
  "phonological": "phonology_wals|phonology_ethnologue",
  "inventory": "inventory_ethnologue|inventory_phoible_aa|inventory_phoible_gm|inventory_phoible_saphon|inventory_phoible_spa|inventory_phoible_ph|inventory_phoible_ra|inventory_phoible_upsid",
  "genetic": "fam",
  "geographic": "geo"
}


def get_languages(indir: str) -> pd.DataFrame:
  df = pd.read_csv(indir, index_col="code", keep_default_na=False)
  df = df.replace({'True': True, 'False': False})
  df = df.astype({'URIEL': bool, 'learned': bool, 'distance': bool})
  df = df[np.where(df["distance"], True, False)]
  return df
  

def get_features(langs: list[str], dtype: str) -> tuple[list[str], np.ndarray]:
  return l2v.get_concatenated_sets(langs, DISTANCE_FEATURE_MAP[dtype])


def get_distances(langs: list[str], dtype: str) -> np.ndarray:
  with zf(l2v.DISTANCES_FILE, 'r') as zp:
    data = sparse.load_npz(zp.open(l2v.map_distance_to_filename(dtype))).todense()
    i_lower, i_upper = np.tril_indices(data.shape[0], -1), np.triu_indices(data.shape[0], 1)
    data[i_lower] = data[i_upper]
    all_indices = [l2v.DISTANCE_LANGUAGES.index(lang) for lang in langs]
    return np.asarray(data[all_indices, :][:, all_indices])


def l2v_dataframe(langs_df: pd.DataFrame, feats: T.Optional[np.ndarray], dists: np.ndarray, dtype: str, outdir: str) -> pd.DataFrame:
  df = langs_df.loc[:, ["URIEL", "joshi", "family"]]
  if feats is not None:
    df["# of NA properties"] = (feats == -1).sum(axis=1)
    df["# of zero properties"] = (feats == 0).sum(axis=1)
    df["# of one properties"] = (feats == 1).sum(axis=1)
    df["mean distance"] = dists.mean(axis=1)
    df["min distance"] = dists.min(axis=1)
    df["Q1 distance"] = np.percentile(dists, 25, axis=1)
    df["median distance"] = np.median(dists, axis=1)
    df["Q3 distance"] = np.percentile(dists, 75, axis=1)
    df["max distance"] = dists.max(axis=1)
    mode = stats.mode(dists, axis=1).mode.flatten()
    df["mode distance"] = mode
    df["# of zero distances"] = (dists == 0).sum(axis=1)
    df["# of mode distances"] = (dists == mode).sum(axis=1)
    df["# of one distances"] = (dists == 1).sum(axis=1)
  df.to_csv(outdir)
  return df


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", type=str)
  parser.add_argument("-o", type=str)
  args = parser.parse_args()

  LANGUAGES = get_languages(args.i)
  
  FEATURES_DATA = {}
  for dtype in l2v.DISTANCES:  
    if dtype in DISTANCE_FEATURE_MAP:
      FEATURES_DATA[dtype] = get_features(list(LANGUAGES.index), dtype)[1]
  
  DISTANCE_DATA = {}
  for dtype in l2v.DISTANCES:
      DISTANCE_DATA[dtype] = get_distances(list(LANGUAGES.index), dtype)

  L2V_DATA = {}
  for dtype in l2v.DISTANCES:
      L2V_DATA[dtype] = l2v_dataframe(LANGUAGES, FEATURES_DATA.get(dtype), DISTANCE_DATA[dtype], dtype, f"{args.o}/{dtype}/{dtype}_data.csv")