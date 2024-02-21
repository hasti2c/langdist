import numpy as np
import pandas as pd
import lang2vec.lang2vec.lang2vec as l2v
from matplotlib import pyplot as plt
import seaborn as sns


def get_l2v_data(dir: str, dtype: str) -> pd.DataFrame:
    df = pd.read_csv(f"{dir}/{dtype}/{dtype}_data.csv", index_col="code", keep_default_na=False)
    df.replace("", np.nan, inplace=True)
    df = df.astype({"joshi": "Float64"}).astype({"joshi": "Int64"})
    return df


def plot_l2v_data(df: pd.DataFrame, dir: str, dtype: str, col: str) -> None:
    sns.histplot(df[col], bins=20)
    plt.savefig(f"{dir}/{dtype}/overall/{dtype}_{col.replace(' ', '_')}.png")
    plt.clf()

    g = sns.histplot(df, x=col, hue="family", multiple="stack", bins=20)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{dir}/{dtype}/by_family/{dtype}_{col.replace(' ', '_')}_by_family.png")
    plt.clf()

    g = sns.histplot(df[df['joshi'] != ''], x=col, hue="joshi", hue_order=["0.0", "1.0", "2.0", "3.0", "4.0", "5.0"], multiple="stack", bins=20)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{dir}/{dtype}/by_joshi/{dtype}_{col.replace(' ', '_')}_by_joshi.png")
    plt.clf()


if __name__ == "__main__":
    L2V_DATA = {}
    for dtype in l2v.DISTANCES:
        L2V_DATA[dtype] = get_l2v_data("top/l2v", dtype)

    for dtype in l2v.DISTANCES:
        plot_l2v_data(L2V_DATA[dtype], "top/l2v", dtype, "# of NA properties")
        plot_l2v_data(L2V_DATA[dtype], "top/l2v", dtype, "# of zero properties")
        plot_l2v_data(L2V_DATA[dtype], "top/l2v", dtype, "# of one properties")
        plot_l2v_data(L2V_DATA[dtype], "top/l2v", dtype, "mean distance")
        plot_l2v_data(L2V_DATA[dtype], "top/l2v", dtype, "min distance")
        plot_l2v_data(L2V_DATA[dtype], "top/l2v", dtype, "Q1 distance")
        plot_l2v_data(L2V_DATA[dtype], "top/l2v", dtype, "median distance")
        plot_l2v_data(L2V_DATA[dtype], "top/l2v", dtype, "Q3 distance")
        plot_l2v_data(L2V_DATA[dtype], "top/l2v", dtype, "max distance")
        plot_l2v_data(L2V_DATA[dtype], "top/l2v", dtype, "mode distance")
        plot_l2v_data(L2V_DATA[dtype], "top/l2v", dtype, "# of zero distances")
        plot_l2v_data(L2V_DATA[dtype], "top/l2v", dtype, "# of mode distances")
        plot_l2v_data(L2V_DATA[dtype], "top/l2v", dtype, "# of one distances")
        