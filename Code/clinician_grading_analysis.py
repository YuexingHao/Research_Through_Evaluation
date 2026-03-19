"""
clinician_grading_analysis.py
------------------------------
Clinician grading analysis: descriptive stats, ICC, bias, and figure generation.

Usage
-----
    python clinician_grading_analysis.py                        # default paths
    python clinician_grading_analysis.py --data-dir /path/to/data --out-dir figures

Dependencies
------------
    pip install pandas openpyxl matplotlib seaborn scipy scikit-learn pingouin statsmodels
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, r2_score

# ── Constants ─────────────────────────────────────────────────────────────────

CRITERIA = ["Completeness", "Correctness", "Clarity", "Empathy"]

C1_COLS = ["Dr.HCompleteness", "Dr.HCorrectness", "Dr.HClarity", "Dr.HEmpathy"]
C2_COLS = ["Dr.BCompleteness", "Dr.BCorrectness", "Dr.BClarity", "Dr.BEmpathy"]
C3_COLS = ["Dr3.Completeness", "Dr3.Correctness", "Dr3.Clarity", "Dr3.Empathy"]

RATER_COLS = {"Dr.H": C1_COLS, "Dr.B": C2_COLS, "Dr3": C3_COLS}
RATER_COL_MAP = {
    "Dr.H": dict(zip(CRITERIA, C1_COLS)),
    "Dr.B": dict(zip(CRITERIA, C2_COLS)),
    "Dr3":  dict(zip(CRITERIA, C3_COLS)),
}

COLOR_BOT  = "#2196F3"
COLOR_TEAM = "#f5a525"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: Path) -> pd.DataFrame:
    df = pd.read_excel(data_dir / "clustered_LLM_outputs.xlsx", engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"DR.BCompleteness": "Dr.BCompleteness"})
    return df


# ── Descriptive statistics ────────────────────────────────────────────────────

def print_descriptive_stats(df: pd.DataFrame):
    gpt_df   = df[df["Type"] == "GPT"]
    human_df = df[df["Type"] == "human"]

    for label, subset in [("All", df), ("GPT", gpt_df), ("Human", human_df)]:
        print(f"\n{'='*60}")
        print(f"Descriptive stats — {label}")
        for rater, cols in RATER_COLS.items():
            print(f"\n  {rater}:")
            print(subset[cols].describe().round(3).to_string())

    comparison = pd.DataFrame({
        "Criteria": CRITERIA,
        **{f"{r}_Mean": df[cols].mean().round(3).values for r, cols in RATER_COLS.items()},
        **{f"{r}_Std":  df[cols].std().round(3).values  for r, cols in RATER_COLS.items()},
    })
    print("\nMean / Std comparison across raters:\n", comparison.to_string(index=False))


# ── ICC ───────────────────────────────────────────────────────────────────────

def _melt_for_icc(df: pd.DataFrame, col_groups: list[list[str]]) -> pd.DataFrame:
    combined = pd.concat([df[cols] for cols in col_groups], axis=1)
    melted = combined.melt(ignore_index=False, var_name="rater", value_name="rating")
    melted["targets"] = melted.index
    return melted


def compute_icc_pairs(df: pd.DataFrame):
    pairs = [("Dr.H", "Dr.B"), ("Dr.H", "Dr3"), ("Dr.B", "Dr3")]
    for r1, r2 in pairs:
        melted = _melt_for_icc(df, [RATER_COLS[r1], RATER_COLS[r2]])
        icc = pg.intraclass_corr(data=melted, targets="targets", raters="rater",
                                  ratings="rating", nan_policy="omit")
        print(f"\nICC — {r1} vs {r2}:\n", icc.to_string(index=False))


def compute_icc_by_category(df: pd.DataFrame, out_dir: Path):
    results = {}
    for criterion in CRITERIA:
        cols = [RATER_COL_MAP[r][criterion] for r in RATER_COL_MAP]
        melted = pd.concat([df[col] for col in cols], axis=1) \
                   .melt(ignore_index=False, var_name="rater", value_name="rating")
        melted["targets"] = melted.index
        icc = pg.intraclass_corr(data=melted, targets="targets", raters="rater",
                                  ratings="rating", nan_policy="omit")
        results[criterion] = icc
        print(f"\nICC by category — {criterion}:\n", icc.to_string(index=False))

    last = list(results.values())[-1]
    out_path = out_dir / "ICC_Results_By_Category.csv"
    last.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


def compute_icc_all_raters(df: pd.DataFrame):
    melted = _melt_for_icc(df, [C1_COLS, C2_COLS, C3_COLS])
    icc = pg.intraclass_corr(data=melted, targets="targets", raters="rater",
                              ratings="rating", nan_policy="omit")
    print("\nICC — all three raters combined:\n", icc.to_string(index=False))


# ── Statistical tests (GPT vs Human per rater) ───────────────────────────────

def compute_gpt_vs_human_stats(df: pd.DataFrame):
    gpt_df   = df[df["Type"] == "GPT"]
    human_df = df[df["Type"] == "human"]
    common   = gpt_df.index.intersection(human_df.index)
    gpt_df, human_df = gpt_df.loc[common], human_df.loc[common]

    print("\nGPT vs Human statistical tests (Dr.H columns):")
    for col, crit in zip(C1_COLS, CRITERIA):
        g, h = gpt_df[col].dropna(), human_df[col].dropna()
        idx  = g.index.intersection(h.index)
        g, h = g.loc[idx], h.loc[idx]
        if len(g) < 2:
            continue
        _, p_wilcox  = stats.wilcoxon(g, h)
        r2           = r2_score(g, h)
        pearson_r, _ = stats.pearsonr(g, h)
        kappa        = cohen_kappa_score(g.astype(int), h.astype(int))
        print(f"  {crit}: Wilcoxon p={p_wilcox:.4f}  R²={r2:.3f}  "
              f"Pearson r={pearson_r:.3f}  Kappa={kappa:.3f}")


# ── Figure helpers ────────────────────────────────────────────────────────────

def _save(fig_or_path: plt.Figure | None, out_path: Path, dpi: int = 180):
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close()


# ── Fig 1: Mean scores bar chart ──────────────────────────────────────────────

def plot_mean_scores_bar(out_dir: Path):
    in_basket_means = [4.48, 4.47, 4.89, 4.50]
    clinical_means  = [4.73, 4.55, 4.92, 4.73]
    std_err_bot     = [0.10, 0.12, 0.08, 0.09]
    std_err_team    = [0.09, 0.10, 0.07, 0.08]
    significant     = [True, False, True, False]

    x = np.arange(len(CRITERIA))
    w = 0.4
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, in_basket_means, w, label="In-Basket Bot",    color=COLOR_BOT,
           yerr=std_err_bot,  error_kw={"elinewidth": 2}, capsize=0)
    ax.bar(x + w/2, clinical_means,  w, label="Clinical Care Team", color=COLOR_TEAM,
           yerr=std_err_team, error_kw={"elinewidth": 2}, capsize=0)

    for i, sig in enumerate(significant):
        if sig:
            h = max(in_basket_means[i] + std_err_bot[i],
                    clinical_means[i]  + std_err_team[i])
            ax.text(x[i], h + 0.05, "*", ha="center", va="bottom", fontsize=20, color="red")

    ax.set_xlabel("Categories", fontsize=18)
    ax.set_ylabel("Mean Scores", fontsize=18)
    ax.set_xticks(x); ax.set_xticklabels(CRITERIA, fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(fontsize=16, loc="lower right")
    plt.tight_layout()
    _save(None, out_dir / "mean_scores_bar.png")


# ── Fig 2: ICC bar chart (hardcoded values) ───────────────────────────────────

def plot_icc_bar(out_dir: Path):
    single_icc1   = [0.211, 0.167, 0.036, 0.217]
    average_icc3k = [0.481, 0.390, 0.170, 0.490]

    x = np.arange(len(CRITERIA))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, single_icc1,   w, label="Single Raters (ICC1)")
    b2 = ax.bar(x + w/2, average_icc3k, w, label="Average Raters (ICC3k)")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")

    ax.set_xlabel("Categories"); ax.set_ylabel("ICC")
    ax.set_xticks(x); ax.set_xticklabels(CRITERIA)
    ax.legend()
    ax.set_title("ICC Comparison by Category")
    plt.tight_layout()
    _save(None, out_dir / "icc_bar.png")


# ── Fig 3: Pairwise bias heatmap + overall bias bar ───────────────────────────

def plot_pairwise_bias(out_dir: Path):
    bias_matrix = pd.DataFrame(
        {"Dr. H": [0, -0.254426, 0.127721],
         "Dr. B": [0.254426, 0, 0.382147],
         "Dr. E": [-0.127721, -0.382147, 0]},
        index=["Dr. H", "Dr. B", "Dr. E"],
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(bias_matrix, annot=True, cmap="coolwarm", fmt=".3f", ax=ax)
    ax.set_title("Pairwise GPT Bias (Completeness)")
    plt.tight_layout()
    _save(None, out_dir / "pairwise_bias_heatmap.png")

    bias_summary = pd.DataFrame({
        "Grader":     ["Dr. H", "Dr. B", "Dr. E"],
        "GPT Bias":   [-0.006574,  0.160048, -0.154975],
        "Human Bias": [-0.003110,  0.124596, -0.120486],
    })
    bias_summary.plot(x="Grader", kind="bar", figsize=(8, 6))
    plt.title("Overall Bias Across Graders")
    plt.ylabel("Mean Bias")
    plt.xticks(rotation=0)
    plt.tight_layout()
    _save(None, out_dir / "overall_bias_bar.png")


# ── Fig 4: Bias radar chart (GPT & Human, 3 graders) ─────────────────────────

def plot_bias_radar(out_dir: Path):
    bias_gpt   = {"C1": [-0.042, 0.000, 0.075, -0.059],
                  "C2": [0.212, 0.066, 0.119, 0.245],
                  "C3": [-0.170, -0.068, -0.194, -0.186]}
    bias_human = {"C1": [-0.058, 0.049, 0.050, -0.055],
                  "C2": [0.183, 0.148, 0.051, 0.116],
                  "C3": [-0.125, -0.197, -0.101, -0.061]}

    labels = np.array(CRITERIA)
    n      = len(CRITERIA)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist() + [0]
    colors = ["blue", "orange", "green"]

    def _radar(data, title, ax):
        for (grader, biases), color in zip(data.items(), colors):
            vals = biases + biases[:1]
            ax.plot(angles, vals, label=grader, linewidth=2, color=color)
            ax.fill(angles, vals, alpha=0.3, color=color)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=16)
        ax.set_yticks([-0.2, 0, 0.2]); ax.set_yticklabels(["-0.2", "0", "0.2"], fontsize=14)
        ax.set_ylim(-0.25, 0.25); ax.set_title(title, size=22, y=1.1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), subplot_kw=dict(polar=True))
    _radar(bias_gpt,   "GPT Bias Across Metrics",   axes[0])
    _radar(bias_human, "Human Bias Across Metrics", axes[1])
    fig.legend(labels=["C1", "C2", "C3"], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), fontsize=16, title="Raters", title_fontsize=18,
               handles=[plt.Line2D([0], [0], color=c, lw=2) for c in colors])
    plt.tight_layout()
    _save(None, out_dir / "bias_radar.png")


# ── Fig 5: Error bar plot — 3 graders (C1, C2, C3) ───────────────────────────

def plot_clinician_errorbar(out_dir: Path):
    np.random.seed(0)
    specs = {
        "c1_bot":  ([4.44, 4.29, 4.88, 4.55], [0.96, 1.10, 0.51, 0.72]),
        "c1_team": ([4.53, 4.64, 4.90, 4.44], [0.70, 0.70, 0.45, 0.81]),
        "c2_bot":  ([4.69, 4.36, 4.93, 4.86], [0.73, 0.99, 0.34, 0.37]),
        "c2_team": ([4.77, 4.74, 4.90, 4.61], [0.63, 0.63, 0.41, 0.70]),
        "c3_bot":  ([4.31, 4.23, 4.62, 4.43], [0.86, 0.93, 0.72, 0.64]),
        "c3_team": ([4.47, 4.40, 4.75, 4.43], [0.83, 0.82, 0.59, 0.65]),
    }
    means = {k: [np.mean(np.random.normal(m, s, 30)) for m, s in zip(*v)] for k, v in specs.items()}
    stds  = {k: [np.std( np.random.normal(m, s, 30)) for m, s in zip(*v)] for k, v in specs.items()}

    colors = {"c1_bot": "#87CEFA", "c1_team": "#fadeb1",
              "c2_bot": "#1E90FF", "c2_team": "#f5a525",
              "c3_bot": "#4682B4", "c3_team": "#FF8C00"}
    labels = {"c1_bot": "C1 (Bot)", "c1_team": "C1 (Team)",
              "c2_bot": "C2 (Bot)", "c2_team": "C2 (Team)",
              "c3_bot": "C3 (Bot)", "c3_team": "C3 (Team)"}
    offsets = {"c1_bot": -0.9, "c1_team": -0.6, "c2_bot": -0.3,
               "c2_team": 0.0, "c3_bot": 0.3,  "c3_team": 0.6}

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    bw = 0.15
    for key in specs:
        for i in range(4):
            xpos = i * 2.0 + offsets[key]
            ax.add_patch(plt.Rectangle((xpos - bw/2, means[key][i] - 0.05),
                                       bw, 0.1, color=colors[key], ec="black"))
            ax.errorbar(xpos, means[key][i], yerr=stds[key][i],
                        fmt="none", ecolor="black", capsize=0, linewidth=1.5)

    ax.set_xticks(np.arange(0, 8, 2))
    ax.set_xticklabels(CRITERIA, fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylim(0, 6)
    ax.legend(handles=[mpatches.Patch(facecolor=colors[k], edgecolor="black", label=labels[k])
                        for k in specs],
              loc="best", bbox_to_anchor=(0.7, 0.32), ncol=3, fontsize=16)
    plt.tight_layout()
    _save(None, out_dir / "clinician_errorbar_3graders.png", dpi=300)


# ── Fig 6: Editing requirements stacked bar (final version with connectors) ───

EDITING_DATA = {
    "Type": ["In-basket bot", "Clinical care team"],
    "Will use this without editing": [79.0, 92.0],
    "Minor (<1 min) editing":        [51.0, 52.0],
    "Major editing":                 [19.0, 11.0],
    "Would not use this":            [ 9.0,  3.0],
}
EDITING_COLS = ["Will use this without editing", "Minor (<1 min) editing",
                "Major editing", "Would not use this"]
COLORS_BOT  = ["#4A90E2", "#6EC2E2", "#A0C4FF", "#D3D3D3"]
COLORS_TEAM = ["#F5A623", "#F7C065", "#FFD57E", "#FFE8B6"]


def _build_editing_df(as_percent: bool = False) -> tuple[pd.DataFrame, list, list]:
    df = pd.DataFrame(EDITING_DATA)
    if as_percent:
        for i in range(2):
            total = df.loc[i, EDITING_COLS].sum()
            df.loc[i, EDITING_COLS] = df.loc[i, EDITING_COLS] / total * 100
    cum_bot  = df.loc[0, EDITING_COLS].cumsum().tolist()
    cum_team = df.loc[1, EDITING_COLS].cumsum().tolist()
    return df, cum_bot, cum_team


def _draw_stacked(ax_or_fig, df, cum_bot, cum_team,
                  bot_label="In-basket bot", team_label="Clinical care team",
                  pct=False, connect_style="--"):
    fig = plt.gcf(); ax = plt.gca()
    for j, (col, cb, ct) in enumerate(zip(EDITING_COLS, COLORS_BOT, COLORS_TEAM)):
        bot_bottom  = df.loc[0, EDITING_COLS[:j]].sum() if j else 0
        team_bottom = df.loc[1, EDITING_COLS[:j]].sum() if j else 0
        ax.bar(bot_label,  df.loc[0, col], bottom=bot_bottom,  color=cb)
        ax.bar(team_label, df.loc[1, col], bottom=team_bottom, color=ct)

    def label(x, y, val, text, fs=15):
        fmt = f"{text} ({val:.1f}%)" if pct else f"{text} ({int(val)})"
        ax.text(x, y, fmt, ha="center", va="center", fontsize=fs, color="black")

    for j, col in enumerate(EDITING_COLS):
        bot_bottom  = df.loc[0, EDITING_COLS[:j]].sum() if j else 0
        team_bottom = df.loc[1, EDITING_COLS[:j]].sum() if j else 0
        label(bot_label,  bot_bottom  + df.loc[0, col] / 2, df.loc[0, col], col)
        label(team_label, team_bottom + df.loc[1, col] / 2, df.loc[1, col], col)

    for i in range(4):
        ax.plot([0, 1], [cum_bot[i], cum_team[i]],
                marker="o", color="#808080", linestyle=connect_style, linewidth=1.5)

    ax.set_ylabel("Number of Responses" if not pct else "Percentage (%)", fontsize=20)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=18)


def plot_editing_stacked(out_dir: Path):
    df, cum_bot, cum_team = _build_editing_df(as_percent=False)
    plt.figure(figsize=(10, 6), dpi=300)
    _draw_stacked(None, df, cum_bot, cum_team)
    plt.tight_layout()
    _save(None, out_dir / "editing_stacked_counts.png", dpi=300)

    df_pct, cum_bot_pct, cum_team_pct = _build_editing_df(as_percent=True)
    plt.figure(figsize=(10, 6), dpi=300)
    _draw_stacked(None, df_pct, cum_bot_pct, cum_team_pct, pct=True)
    plt.tight_layout()
    _save(None, out_dir / "editing_stacked_percent.png", dpi=300)


# ── Fig 7: Multi-round bar chart (Rounds 1–4) ─────────────────────────────────

def plot_multi_round_bar(out_dir: Path):
    from matplotlib.patches import Patch

    # Round 1
    r1_C1_m, r1_C1_s = [2.59, 2.56, 2.59, 2.74], [0.84, 0.97, 0.93, 0.53]
    r1_C2_m, r1_C2_s = [2.67, 2.33, 2.85, 2.48], [0.92, 0.68, 0.72, 0.80]
    # Round 2
    r2_C1b_m, r2_C1b_s = [3.60, 3.20, 4.10, 3.00], [1.26, 1.48, 0.88, 1.05]
    r2_C1t_m, r2_C1t_s = [4.47, 4.68, 4.89, 4.16], [0.77, 0.48, 0.32, 0.60]
    r2_C2b_m, r2_C2b_s = [4.70, 4.30, 4.90, 4.70], [0.95, 1.06, 0.32, 0.95]
    r2_C2t_m, r2_C2t_s = [4.84, 4.47, 4.95, 4.53], [0.37, 0.96, 0.23, 0.70]
    # Round 3
    r3_C1b_m, r3_C1b_s = [4.44, 4.29, 4.88, 4.55], [0.96, 1.10, 0.51, 0.72]
    r3_C1t_m, r3_C1t_s = [4.53, 4.64, 4.90, 4.44], [0.70, 0.70, 0.45, 0.81]
    r3_C2b_m, r3_C2b_s = [4.69, 4.36, 4.93, 4.86], [0.73, 0.99, 0.34, 0.37]
    r3_C2t_m, r3_C2t_s = [4.77, 4.74, 4.90, 4.61], [0.63, 0.63, 0.41, 0.70]
    # Round 4 (LLM graders)
    r4_gpt4o_h_m, r4_gpt4o_h_s = [3.38, 4.72, 4.49, 3.24], [0.91, 0.58, 0.73, 1.19]
    r4_gem_h_m,   r4_gem_h_s   = [2.78, 3.77, 3.81, 2.78], [0.88, 0.87, 0.88, 0.98]
    r4_gpt4o_b_m, r4_gpt4o_b_s = [3.89, 4.89, 4.82, 4.11], [0.84, 0.40, 0.43, 0.98]
    r4_gem_b_m,   r4_gem_b_s   = [3.23, 4.23, 4.31, 3.28], [0.89, 0.70, 0.64, 0.90]

    cats = (["Complete", "Correct", "Concise", "Trustworthy"] +
            ["Complete", "Correct", "Clarity",  "Empathy"]    * 3)
    x = np.arange(len(cats)); bw = 0.12

    c_lo = "#fadeb1"; c_hi = "#f5a525"
    b_lo = "#87CEFA"; b_hi = "#1E90FF"
    c_pur_lo = "#d6a5e1"; c_pur = "#9b59b6"
    c_grn_lo = "#ACE1AF"; c_grn = "#17B169"

    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    kw = dict(capsize=0)

    def bars(xslice, m, s, color): ax.bar(xslice, m, bw, yerr=s, color=color, **kw)

    bars(x[:4] - bw,   r1_C1_m,    r1_C1_s,    c_lo)
    bars(x[:4],        r1_C2_m,    r1_C2_s,    c_hi)
    bars(x[4:8] - bw,  r2_C1b_m,   r2_C1b_s,   b_lo)
    bars(x[4:8],       r2_C1t_m,   r2_C1t_s,   c_lo)
    bars(x[4:8] + bw,  r2_C2b_m,   r2_C2b_s,   b_hi)
    bars(x[4:8] + 2*bw,r2_C2t_m,   r2_C2t_s,   c_hi)
    bars(x[8:12] - bw, r3_C1b_m,   r3_C1b_s,   b_lo)
    bars(x[8:12],      r3_C1t_m,   r3_C1t_s,   c_lo)
    bars(x[8:12] + bw, r3_C2b_m,   r3_C2b_s,   b_hi)
    bars(x[8:12]+2*bw, r3_C2t_m,   r3_C2t_s,   c_hi)
    bars(x[12:16]-2*bw,r4_gpt4o_h_m,r4_gpt4o_h_s,c_pur_lo)
    bars(x[12:16]- bw, r4_gem_h_m,  r4_gem_h_s,  c_grn_lo)
    bars(x[12:16],     r4_gpt4o_b_m,r4_gpt4o_b_s,c_pur)
    bars(x[12:16]+ bw, r4_gem_b_m,  r4_gem_b_s,  c_grn)

    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=12)
    ax.set_ylabel("Mean Scores", fontsize=18)
    ax.axvline(3.5, color="black", linestyle="--", lw=1)
    ax.axvline(7.5, color="black", linestyle="--", lw=1)
    ax.axvline(11.5, color="black", linestyle="--", lw=1)

    ax2 = ax.twiny()
    ax2.set_xticks([2, 6, 10, 14])
    ax2.set_xticklabels(["Round 1 (30 pairs)", "Round 2 (15 pairs)",
                          "Round 3 (158 pairs)", "Round 4 (LLM graders)"], fontsize=14)
    ax2.set_xlim(ax.get_xlim())

    legend_patches = [
        Patch(facecolor=b_lo,      label="C1 (In-basket Bot)"),
        Patch(facecolor=b_hi,      label="C2 (In-basket Bot)"),
        Patch(facecolor=c_lo,      label="C1 (Clinical Care Team)"),
        Patch(facecolor=c_hi,      label="C2 (Clinical Care Team)"),
        Patch(facecolor=c_pur_lo,  label="GPT-4o (Clinical Care Team)"),
        Patch(facecolor=c_grn_lo,  label="Gemini (Clinical Care Team)"),
        Patch(facecolor=c_pur,     label="GPT-4o (In-basket Bot)"),
        Patch(facecolor=c_grn,     label="Gemini (In-basket Bot)"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=13)
    plt.tight_layout()
    _save(None, out_dir / "multi_round_bar.png", dpi=300)


# ── Fig 8: Round 5 — LLM grader scatter plot ─────────────────────────────────

def plot_llm_grader_scatter(out_dir: Path):
    human_means = {
        "Llama":         [3.032, 4.253, 3.873, 2.418],
        "GPT-4o":        [3.38,  4.72,  4.49,  3.24 ],
        "Gemini":        [2.78,  3.77,  3.81,  2.78 ],
        "GPT-4":         [3.994, 4.807, 4.876, 4.025],
        "Clinician Avg": [4.59,  4.59,  4.85,  4.49 ],
    }
    human_stds = {
        "Llama":         [0.85, 0.78, 0.72, 0.68],
        "GPT-4o":        [0.91, 0.58, 0.73, 1.19],
        "Gemini":        [0.88, 0.87, 0.88, 0.98],
        "GPT-4":         [1.08, 0.63, 0.38, 1.23],
        "Clinician Avg": [0.16, 0.18, 0.09, 0.10],
    }
    bot_means = {
        "Llama":         [3.509, 4.623, 4.340, 3.031],
        "GPT-4o":        [3.89,  4.89,  4.82,  4.11 ],
        "Gemini":        [3.23,  4.23,  4.31,  3.28 ],
        "GPT-4":         [4.679, 4.962, 4.969, 4.843],
        "Clinician Avg": [4.48,  4.29,  4.81,  4.61 ],
    }
    bot_stds = {
        "Llama":         [0.90, 0.70, 0.75, 0.77],
        "GPT-4o":        [0.84, 0.40, 0.43, 0.98],
        "Gemini":        [0.89, 0.70, 0.64, 0.90],
        "GPT-4":         [0.71, 0.22, 0.18, 0.53],
        "Clinician Avg": [0.19, 0.06, 0.16, 0.22],
    }

    color_h = {"Llama": "#b2d8b2", "GPT-4o": "#d6a5e1",
               "Gemini": "#FFFF00", "GPT-4": "#85c1e9", "Clinician Avg": "#ffa07a"}
    color_b = {"Llama": "#76b947", "GPT-4o": "#9b59b6",
               "Gemini": "#17B169", "GPT-4": "#3498db", "Clinician Avg": "#ff6347"}

    x = np.arange(len(CRITERIA))
    bw = 0.10
    offsets_h = np.linspace(-2.4, -0.6, 5) * bw
    offsets_b = np.linspace( 0.0,  2.4, 5) * bw

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    for i, key in enumerate(human_means):
        ax.errorbar(x + offsets_h[i], human_means[key], yerr=human_stds[key],
                    fmt="none", ecolor="black", elinewidth=1.5)
        ax.scatter(x + offsets_h[i], human_means[key], s=60, marker="s",
                   color=color_h[key], label=f"{key} (Clinical Care Team)", edgecolor="black")
        ax.errorbar(x + offsets_b[i], bot_means[key],   yerr=bot_stds[key],
                    fmt="none", ecolor="black", elinewidth=1.5)
        ax.scatter(x + offsets_b[i], bot_means[key], s=60, marker="s",
                   color=color_b[key], label=f"{key} (In-basket Bot)", edgecolor="black")

    ax.set_xticks(x); ax.set_xticklabels(CRITERIA, fontsize=18)
    ax.set_ylim(0, 5.5); ax.set_ylabel("Mean Scores", fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.15), ncol=2, fontsize=13)
    plt.tight_layout()
    _save(None, out_dir / "llm_grader_scatter.png", dpi=300)


# ── Fig 9: Trends line plot ───────────────────────────────────────────────────

def plot_trends_line(out_dir: Path):
    x = np.arange(4)
    r3_C1 = [4.53, 4.64, 4.90, 4.44]
    r3_C2 = [4.77, 4.74, 4.90, 4.61]
    r4_gpt4o = [3.38, 4.72, 4.49, 3.24]
    r4_gemini = [2.78, 3.77, 3.81, 2.78]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, r3_C1,    label="C1 (Round 3)",       marker="^")
    ax.plot(x, r3_C2,    label="C2 (Round 3)",       marker="^")
    ax.plot(x, r4_gpt4o, label="GPT-4o (Round 4)",   marker="D")
    ax.plot(x, r4_gemini,label="Gemini (Round 4)",    marker="D")

    ax.set_xticks(x); ax.set_xticklabels(CRITERIA)
    ax.set_xlabel("Category"); ax.set_ylabel("Mean Scores")
    ax.set_title("Trends Across Rounds — C1, C2 vs LLM Graders")
    ax.legend(loc="best")
    plt.tight_layout()
    _save(None, out_dir / "trends_line.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description="Clinician grading analysis and figure generation.")
    p.add_argument("--data-dir", type=Path, default=here,
                   help="Directory containing clustered_LLM_outputs.xlsx (default: script dir)")
    p.add_argument("--out-dir",  type=Path, default=here / "output",
                   help="Directory for output figures and CSV (default: Code/output/)")
    p.add_argument("--skip-icc", action="store_true",
                   help="Skip ICC computation (slow for large datasets)")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    df = load_data(args.data_dir)

    print("\nDescriptive statistics…")
    print_descriptive_stats(df)

    if not args.skip_icc:
        print("\nICC between rater pairs…")
        compute_icc_pairs(df)

        print("\nICC by criterion…")
        compute_icc_by_category(df, args.out_dir)

        print("\nICC across all raters…")
        compute_icc_all_raters(df)

    print("\nGPT vs Human statistical tests…")
    compute_gpt_vs_human_stats(df)

    print("\nGenerating figures…")
    plot_mean_scores_bar(args.out_dir)
    plot_icc_bar(args.out_dir)
    plot_pairwise_bias(args.out_dir)
    plot_bias_radar(args.out_dir)
    plot_clinician_errorbar(args.out_dir)
    plot_editing_stacked(args.out_dir)
    plot_multi_round_bar(args.out_dir)
    plot_llm_grader_scatter(args.out_dir)
    plot_trends_line(args.out_dir)

    print(f"\nAll done. Output saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
