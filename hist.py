import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_chain(path: str) -> pd.DataFrame:
    with open(path) as fh:
        first = fh.readline().strip()
    if first.startswith("#"):
        cols = first.lstrip("#").split()
    else:
        cols = None

    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=cols)
    return df


def load_raw(path: str) -> pd.DataFrame:
    with open(path) as fh:
        first = fh.readline().strip()
    cols = first.lstrip("#").split()

    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=cols)
    return df


def filter_bad(df: pd.DataFrame) -> pd.DataFrame:
    if "Yp" not in df.columns:
        return df
    return df[
        (df["Yp"] < 0.250)
        & (df["Yp"] > 0.240)
        & (df["DoHx1e5"] > 2.4)
        & (df["DoHx1e5"] < 2.56)
        & (df["He3oHx1e5"] > 1.0452)
        & (df["Li7oHx1e10"] > 5)
    ].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Plot MCMC posterior histograms")
    parser.add_argument(
        "csv",
        nargs="?",
        default="chain_ede_Linear_2026-03-04_01-46-31.csv",
        help="CSV file to plot (chain_* preferred)",
    )
    parser.add_argument(
        "--burnin",
        type=int,
        default=0,
        help="Extra burn-in rows to skip from the *raw* CSV "
        "(chain_* files already have burn-in removed)",
    )
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    path = args.csv
    is_chain = path.startswith("chain_")

    print(f"Loading: {path}  (is_chain={is_chain})")
    if is_chain:
        df = load_chain(path)
    else:
        df = load_raw(path)
        df = df.iloc[args.burnin :]
        n_before = len(df)
        df = filter_bad(df)
        print(
            f"  Filtered {n_before - len(df)} bad rows (Yp outside (0,1)), {len(df)} remain"
        )

    ncols = len(df.columns)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    lower_bound = df.iloc[:, 0].quantile(0.025)
    upper_bound = df.iloc[:, 0].quantile(0.975)
    filtered_data = df[(df.iloc[:, 0] >= lower_bound) & (df.iloc[:, 0] <= upper_bound)]

    for ax, col in zip(axes, df.columns):
        sns.kdeplot(data=filtered_data, x=col, fill=True, ax=ax, color="steelblue")
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(col, fontsize=12)
        ax.grid(False)

    plt.tight_layout()
    plt.suptitle(f"Posterior Distributions — {path}", y=1.02, fontsize=11)
    plt.show()


if __name__ == "__main__":
    main()
