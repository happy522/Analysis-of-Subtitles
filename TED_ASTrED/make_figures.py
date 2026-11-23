"""Figures for TED / ASTrED (direct vs indirect German subtitles), in the style of Subtitles_analysis/.
Reads sentence_scores.csv -> figures/<metric>_<plot>.png"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t as tdist, wilcoxon

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
for old in FIG.glob("*.png"):
    old.unlink()

df = pd.read_csv(HERE / "sentence_scores.csv")
v = df[~df["excluded"]].copy()
v["German target"] = v["direction"].map({"direct": "DE direct", "indirect": "DE indirect"})

SRC = ["ES source", "EN pivot", "EN_CC pivot"]
EPS = ["S01E02", "S02E06", "S02E08", "S02E10", "S03E07"]
HUE = ["DE direct", "DE indirect"]
PAL = {"DE direct": "tab:blue", "DE indirect": "tab:orange"}
DPI = 300


def ci95(x):
    x = np.asarray(x, float)
    return tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=DPI)
    plt.close(fig)
    print("saved", FIG / name)


for m in ["TED", "ASTrED"]:
    # 1. Mean per setting (bar + 95% CI)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=v, x="source", y=m, hue="German target", order=SRC, hue_order=HUE, palette=PAL,
                errorbar=("ci", 95), n_boot=2000, seed=0, capsize=0.08, err_kws={"linewidth": 1.2}, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=10, fontsize=9)
    ax.set_title(f"Mean {m} per sentence: source/pivot vs. German subtitles")
    ax.set_xlabel("Source / pivot")
    ax.set_ylabel(f"Mean {m}")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.legend(title="German target", loc="upper right")
    save(fig, f"{m}_mean_barplot.png")

    # 2. Boxplot of sentence-level scores
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=v, x="source", y=m, hue="German target", order=SRC, hue_order=HUE, palette=PAL, ax=ax)
    ax.set_title(m)
    ax.set_xlabel("Source / pivot")
    ax.set_ylabel(m)
    ax.set_ylim(top=v[m].max() * 1.3)
    ax.legend(title="German target", loc="upper right", ncol=2)
    save(fig, f"{m}_boxplot.png")

    # 3. KDE of sentence-level scores, one panel per source/pivot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, s in zip(axes, SRC):
        for h in HUE:
            sns.kdeplot(v[(v.source == s) & (v["German target"] == h)][m], label=h, fill=True, alpha=0.4,
                        color=PAL[h], bw_adjust=1.5, clip=(0, None), ax=ax)
        ax.set_title(s)
        ax.set_xlabel(m)
    axes[0].legend()
    fig.suptitle(f"KDE of {m} by German target")
    save(fig, f"{m}_KDE.png")

    # 4. Paired difference direct - indirect (mean + 95% CI)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, s in enumerate(SRC):
        piv = v[v.source == s].pivot_table(index=["episode", "sent_id"], columns="direction", values=m).dropna()
        diff = piv["direct"] - piv["indirect"]
        p = wilcoxon(piv["direct"], piv["indirect"]).pvalue
        ax.errorbar(diff.mean(), i, xerr=ci95(diff), fmt="o", color="tab:blue", capsize=4, markersize=7)
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.annotate(f"{diff.mean():+.2f} ({stars})", (diff.mean(), i), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(SRC)), SRC)
    ax.set_ylim(-0.6, len(SRC) - 0.4)
    ax.invert_yaxis()
    ax.set_title(f"Paired difference in {m}: DE direct − DE indirect")
    ax.set_xlabel(f"Mean difference in {m} (95% CI); Wilcoxon: * p<.05, ** p<.01, *** p<.001")
    ax.set_ylabel("Source / pivot")
    save(fig, f"{m}_paired_difference.png")

    # 5. Per episode
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, s in zip(axes, SRC):
        for h in HUE:
            g = v[(v.source == s) & (v["German target"] == h)].groupby("episode")[m]
            ax.errorbar(EPS, [g.get_group(e).mean() for e in EPS], yerr=[ci95(g.get_group(e)) for e in EPS],
                        marker="o", capsize=3, label=h, color=PAL[h])
        ax.set_title(s)
        ax.set_xlabel("Episode")
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel(f"Mean {m}")
    axes[0].legend()
    fig.suptitle(f"Mean {m} per episode (95% CI)")
    save(fig, f"{m}_per_episode.png")

# ------------------------------------------------ 6. Final summary table (image)
NAMES = {"ES source": "Spanish source", "EN pivot": "English pivot", "EN_CC pivot": "English CC pivot"}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


main_rows, comp_rows = [], []
for sid, (s, h) in enumerate([(s, h) for s in SRC for h in HUE], 1):
    d = v[(v.source == s) & (v["German target"] == h)]
    main_rows.append([f"({sid})", NAMES[s], h.replace("DE", "German"), f"{len(d):,}",
                      f"{d.TED.mean():.2f} ± {d.TED.std():.2f}", f"{d.ASTrED.mean():.2f} ± {d.ASTrED.std():.2f}"])
for s in SRC:
    piv = v[v.source == s].pivot_table(index=["episode", "sent_id"], columns="direction",
                                       values=["TED", "ASTrED"]).dropna()
    cells = [NAMES[s], f"{len(piv):,}"]
    for m in ["TED", "ASTrED"]:
        diff = piv[(m, "direct")] - piv[(m, "indirect")]
        p = wilcoxon(piv[(m, "direct")], piv[(m, "indirect")]).pvalue
        cells.append(f"{diff.mean():+.2f} {stars(p)}")
        cells.append("< .001" if p < 0.001 else f"{p:.3f}")
    comp_rows.append(cells)


def draw_table(ax, header, rows, col_widths, title, shade_groups_of=None):
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header, colWidths=col_widths, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bfbfbf")
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e6e6e6")
        elif shade_groups_of and ((r - 1) // shade_groups_of) % 2 == 1:
            cell.set_facecolor("#f5f5f5")
    ax.set_title(title, pad=4)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.2), gridspec_kw={"height_ratios": [7, 4.6], "hspace": 0.05})
draw_table(ax1, ["", "Source / pivot", "German target", "N pairs", "TED (mean ± SD)", "ASTrED (mean ± SD)"],
           main_rows, [0.06, 0.2, 0.18, 0.11, 0.2, 0.2],
           "TED and ASTrED per sentence: source/pivot vs. German subtitles", shade_groups_of=2)
draw_table(ax2, ["Source / pivot", "Paired N", "Δ TED", "p (TED)", "Δ ASTrED", "p (ASTrED)"],
           comp_rows, [0.2, 0.11, 0.15, 0.12, 0.15, 0.12],
           "Direct − indirect (paired, same sentences; Wilcoxon signed-rank)")
fig.text(0.5, 0.13, "Higher = more syntactic divergence from the source/pivot.  Δ > 0: direct German diverges more.  "
         "* p<.05, ** p<.01, *** p<.001", ha="center", fontsize=9)
save(fig, "summary_table.png")
