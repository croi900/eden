#!/usr/bin/env python

import sys
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Peek at running MCMC/NS samples sorted by logL"
    )
    parser.add_argument("file", help="Path to the samples CSV file (e.g. ns_CC_...csv)")
    parser.add_argument(
        "--top", type=int, default=1000, help="Number of rows to show (default = 200)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save a histogram of the main parameter",
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file, sep=" ", comment="#", header=None)

        with open(args.file, "r") as f:
            first_line = f.readline().strip()

        if first_line.startswith("#"):
            columns = first_line.lstrip("#").split()

            if len(columns) == df.shape[1]:
                df.columns = columns
            else:
                print(
                    f"Warning: header has {len(columns)} columns but data has {df.shape[1]}"
                )

    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

    if df.empty:
        print(f"File {args.file} is currently empty.")
        sys.exit(0)

    logl_col = "logL" if "logL" in df.columns else df.columns[-1]

    df = df.dropna(subset=[logl_col])

    df_sorted = df.sort_values(by=logl_col, ascending=False)

    N_BEST = 8000
    df_top = df_sorted.head(N_BEST).reset_index(drop=True)

    print(f"\nCurrently tracking {len(df)} valid evaluations.\n")
    print(f"Top {args.top} Best Log-Likelihoods:")
    print("-" * 60)

    print(df_top.head(args.top).to_string())
    print("\n" + "=" * 60)

    param_col = df_top.columns[0]
    mean_val = df_top[param_col].mean()
    median_val = df_top[param_col].median()
    std_val = df_top[param_col].std()
    min_val = df_top[param_col].min()
    max_val = df_top[param_col].max()

    import numpy as np

    upper_95 = np.percentile(df_top[param_col], 95)

    print(f"Stats for {param_col} (from top {min(len(df_top), N_BEST)} best samples):")
    print(f"  Mean   = {mean_val:.4e}")
    print(f"  Median = {median_val:.4e}")
    print(f"  StdDev = {std_val:.4e}")
    print(f"  Range  = [{min_val:.4e}, {max_val:.4e}]")
    print(f"  --------------------------")
    print(f"  95% UL = {upper_95:.4e}\n")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import math
        except ImportError:
            print(
                "Error: matplotlib and seaborn are required for plotting. Install with 'uv add matplotlib seaborn'"
            )
            sys.exit(1)

        columns_to_plot = [c for c in df_top.columns if c != logl_col]
        num_cols = len(columns_to_plot)
        grid_cols = 3
        grid_rows = math.ceil(num_cols / grid_cols)

        plt.figure(figsize=(15, 4 * grid_rows))

        for i, col in enumerate(columns_to_plot):
            plt.subplot(grid_rows, grid_cols, i + 1)

            c_min = df_top[col].min()
            c_max = df_top[col].max()
            c_median = df_top[col].median()
            c_upper95 = np.percentile(df_top[col], 95)

            is_log_scale = False
            if c_min > 0 and (c_max / c_min > 100):
                is_log_scale = True

            sns.histplot(
                df_top[col],
                bins=30,
                kde=True,
                color="blue",
                alpha=0.6,
                log_scale=is_log_scale,
            )

            plt.axvline(
                c_median,
                color="green",
                linestyle="dashed",
                linewidth=1.5,
                label=f"Median ({c_median:.2e})",
            )
            if col == param_col:
                plt.axvline(
                    c_upper95,
                    color="red",
                    linestyle="dashed",
                    linewidth=1.5,
                    label=f"95% UL ({c_upper95:.2e})",
                )

            title_suffix = " (Log Scale)" if is_log_scale else ""
            plt.title(f"{col}{title_suffix}")
            plt.xlabel(col)
            plt.ylabel("")
            plt.legend(fontsize="small")

        plt.suptitle(
            f"Distributions from Top {min(len(df_top), N_BEST)} LogL Samples",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # ty:ignore[invalid-argument-type]

        out_plot = "peek_all_distributions.png"
        plt.savefig(out_plot, dpi=150)
        print(f"Histograms grid saved to: {out_plot}")


if __name__ == "__main__":
    main()
