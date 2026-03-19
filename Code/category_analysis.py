"""
category_analysis.py
--------------------
Analyse per-category grading performance and generate figures.

Usage
-----
    python category_analysis.py                         # use default paths
    python category_analysis.py --data-dir /path/to/data --out-dir figures

Dependencies
------------
    pip install pandas openpyxl matplotlib seaborn wordcloud nltk
    python -m nltk.downloader punkt averaged_perceptron_tagger
"""

import argparse
import os
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# ── Constants ────────────────────────────────────────────────────────────────

CATEGORY_MAPPING = {
    1: "Test Results",
    2: "Side Effects",
    3: "Medication Question",
    4: "Radiation Treatment Question",
    5: "Medical Oncology Question",
    6: "Surgical Oncology Question",
    7: "Care Coordination/Logistics",
    8: "Lab/Radiology/Pathology Reports",
    9: "Care Journey Questions",
}

COLOR_HUMAN = "#f5a525"
COLOR_GPT   = "#2196F3"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    each_category_df = pd.read_excel(data_dir / "Each_Category_Analysis.xlsx",      engine="openpyxl")
    clustered_llm_df = pd.read_excel(data_dir / "clustered_LLM_outputs.xlsx",       engine="openpyxl")
    round4_df        = pd.read_excel(data_dir / "Round4_InBasket_Grading_MDs.xlsx", engine="openpyxl")
    return each_category_df, clustered_llm_df, round4_df


# ── Merge & save ─────────────────────────────────────────────────────────────

def merge_and_save(clustered_llm_df: pd.DataFrame,
                   each_category_df: pd.DataFrame,
                   out_dir: Path) -> pd.DataFrame:
    merged = pd.merge(
        clustered_llm_df,
        each_category_df[["prompt", "Category Number"]],
        on="prompt",
        how="outer",
        suffixes=("", "_new"),
    )
    if "Category Number_new" in merged.columns:
        merged["Category Number"] = merged["Category Number_new"].combine_first(merged["Category Number"])
        merged.drop(columns=["Category Number_new"], inplace=True)

    out_path = out_dir / "Updated_clustered_LLM_outputs.xlsx"
    merged.to_excel(out_path, index=False, engine="openpyxl")
    print(f"Saved merged file → {out_path}")
    return merged


# ── Per-category means ────────────────────────────────────────────────────────

def compute_means(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    human_df = df[df["Type"] == "human"]
    gpt_df   = df[df["Type"] == "GPT"]
    human_means = human_df.groupby("Category Number")["HBTotal"].mean().reset_index()
    gpt_means   = gpt_df.groupby("Category Number")["HBTotal"].mean().reset_index()
    print("Mean HBTotal – Human:\n", human_means.to_string(index=False))
    print("\nMean HBTotal – GPT:\n",  gpt_means.to_string(index=False))
    return human_means, gpt_means


# ── Radar chart ───────────────────────────────────────────────────────────────

def plot_radar(human_means: pd.DataFrame, gpt_means: pd.DataFrame, out_dir: Path):
    labels = list(CATEGORY_MAPPING.values())
    n = len(labels)

    # Align values to CATEGORY_MAPPING order
    h_vals = [human_means.set_index("Category Number")["HBTotal"].get(k, 0) for k in CATEGORY_MAPPING]
    g_vals = [gpt_means.set_index("Category Number")["HBTotal"].get(k, 0)   for k in CATEGORY_MAPPING]

    angles = [i / n * 2 * pi for i in range(n)] + [0]
    h_vals = h_vals + [h_vals[0]]
    g_vals = g_vals + [g_vals[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, h_vals, linewidth=2, color=COLOR_HUMAN, label="Human Care Team")
    ax.fill(angles, h_vals, color=COLOR_HUMAN, alpha=0.3)
    ax.plot(angles, g_vals, linewidth=2, color=COLOR_GPT,   label="RadOnc-GPT")
    ax.fill(angles, g_vals, color=COLOR_GPT,   alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_rlabel_position(30)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_yticklabels(["0", "5", "10", "15", "20"], color="grey", size=8)
    ax.set_ylim(0, 20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.36, 1.1))
    plt.tight_layout()

    out_path = out_dir / "radar_chart.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


# ── Bar chart ─────────────────────────────────────────────────────────────────

def plot_bar(df: pd.DataFrame, out_dir: Path):
    df = df.copy()
    df["Category Name"] = df["Category Number"].map(CATEGORY_MAPPING)

    human_means = df[df["Type"] == "human"].groupby("Category Name")["HBTotal"].mean().reset_index()
    gpt_means   = df[df["Type"] == "GPT"].groupby("Category Name")["HBTotal"].mean().reset_index()
    human_means["Type"] = "Human"
    gpt_means["Type"]   = "GPT"
    combined = pd.concat([human_means, gpt_means])

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="Category Name", y="HBTotal", hue="Type", data=combined,
                palette={"Human": COLOR_HUMAN, "GPT": COLOR_GPT}, ax=ax)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 4),
                        textcoords="offset points")
    ax.set_xlabel("Category", fontsize=13)
    ax.set_ylabel("Mean Grader Total", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = out_dir / "bar_chart.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


# ── Donut chart ───────────────────────────────────────────────────────────────

def plot_donut(df: pd.DataFrame, out_dir: Path, color_coded: bool = True):
    df = df.copy()
    df["Category Name"] = df["Category Number"].map(CATEGORY_MAPPING)

    human_means = df[df["Type"] == "human"].groupby("Category Name")["HBTotal"].mean().reset_index()
    gpt_means   = df[df["Type"] == "GPT"].groupby("Category Name")["HBTotal"].mean().reset_index()
    combined    = pd.merge(human_means, gpt_means, on="Category Name", suffixes=("_Human", "_GPT"))
    cat_counts  = df["Category Name"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        cat_counts,
        labels=cat_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=plt.cm.Paired.colors,
        explode=(0.05,) * len(cat_counts),
        pctdistance=0.85,
        wedgeprops=dict(width=0.3),
    )
    for text in texts:      text.set_fontsize(13)
    for autotext in autotexts: autotext.set_fontsize(11)

    for i, p in enumerate(wedges):
        cat = cat_counts.index[i]
        row = combined[combined["Category Name"] == cat]
        if row.empty:
            continue
        h_val = row["HBTotal_Human"].values[0]
        g_val = row["HBTotal_GPT"].values[0]

        angle = (p.theta2 - p.theta1) / 2 + p.theta1
        x, y  = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        ha    = "left" if x >= 0 else "right"
        face  = (COLOR_GPT if g_val > h_val else COLOR_HUMAN) if color_coded else "white"

        ax.annotate(
            f"Human: {h_val:.2f}\nGPT: {g_val:.2f}",
            xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
            textcoords="data", ha=ha, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor=face),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="black"),
        )

    plt.axis("equal")
    plt.tight_layout()

    suffix   = "color_coded" if color_coded else "plain"
    out_path = out_dir / f"donut_chart_{suffix}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


# ── Word cloud ────────────────────────────────────────────────────────────────

def plot_wordcloud(round4_df: pd.DataFrame, out_dir: Path):
    nltk.download("punkt",                    quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("punkt_tab",                quiet=True)

    comments = " ".join(round4_df["Comments"].dropna().tolist())
    tokens   = word_tokenize(comments)
    tagged   = pos_tag(tokens)
    filtered = " ".join(w for w, pos in tagged if pos in ("NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"))

    wc = WordCloud(width=800, height=400, background_color="white").generate(filtered)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()

    out_path = out_dir / "wordcloud.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Category analysis and figure generation.")
    parser.add_argument("--data-dir", type=Path, default=here,
                        help="Directory containing the three input Excel files (default: script directory)")
    parser.add_argument("--out-dir",  type=Path, default=here / "output",
                        help="Directory where figures and merged Excel are saved (default: Code/output/)")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    each_category_df, clustered_llm_df, round4_df = load_data(args.data_dir)

    print("\nMerging category numbers and saving…")
    clustered_llm_df = merge_and_save(clustered_llm_df, each_category_df, args.out_dir)

    print("\nComputing per-category means…")
    human_means, gpt_means = compute_means(clustered_llm_df)

    print("\nGenerating figures…")
    plot_radar(human_means, gpt_means, args.out_dir)
    plot_bar(clustered_llm_df, args.out_dir)
    plot_donut(clustered_llm_df, args.out_dir, color_coded=False)
    plot_donut(clustered_llm_df, args.out_dir, color_coded=True)
    plot_wordcloud(round4_df, args.out_dir)

    print(f"\nAll done. Figures saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
