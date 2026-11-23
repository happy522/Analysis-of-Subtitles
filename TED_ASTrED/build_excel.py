"""Build TED_ASTrED/TED_ASTrED_results.xlsx from sentence_scores.csv."""
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "sentence_scores.csv")
valid = df[~df["excluded"]].copy()
OUT = HERE / "TED_ASTrED_results.xlsx"

SRC_ORDER = ["ES source", "EN pivot", "EN_CC pivot"]
METRICS = ["TED", "ASTrED", "TED_norm", "ASTrED_norm"]

# ---------------------------------------------------------------- 1. Summary (6 settings)
g = valid.groupby(["setting", "source", "target"])
summary = g.agg(
    N_pairs=("TED", "size"),
    TED_mean=("TED", "mean"), TED_sd=("TED", "std"), TED_median=("TED", "median"),
    TED_norm_mean=("TED_norm", "mean"),
    ASTrED_mean=("ASTrED", "mean"), ASTrED_sd=("ASTrED", "std"), ASTrED_median=("ASTrED", "median"),
    ASTrED_norm_mean=("ASTrED_norm", "mean"),
    TED_total=("TED", "sum"), ASTrED_total=("ASTrED", "sum"),
).reset_index()
excl = df[df["excluded"]].groupby("setting").size()
summary.insert(4, "Excluded_pairs", summary["setting"].map(excl).fillna(0).astype(int))
summary.columns = ["Setting", "Source / Pivot", "German target", "N pairs", "Excluded (empty '-')",
                   "TED mean", "TED SD", "TED median", "TED norm. mean",
                   "ASTrED mean", "ASTrED SD", "ASTrED median", "ASTrED norm. mean",
                   "TED total", "ASTrED total"]

# ---------------------------------------------------------------- 2. Overview matrix
matrix_rows = []
for src in SRC_ORDER:
    r = {"Source / Pivot": src}
    for m in ["TED", "ASTrED"]:
        for d in ["direct", "indirect"]:
            r[f"{m} – DE {d}"] = valid[(valid.source == src) & (valid.direction == d)][m].mean()
    matrix_rows.append(r)
matrix = pd.DataFrame(matrix_rows)

# ---------------------------------------------------------------- 3. Direct vs indirect (paired)
comp_rows = []
for src in SRC_ORDER:
    sub = valid[valid.source == src]
    piv = sub.pivot_table(index=["episode", "sent_id"], columns="direction", values=METRICS)
    piv = piv.dropna()
    for m in METRICS:
        d, i = piv[(m, "direct")], piv[(m, "indirect")]
        diff = d - i
        try:
            p = wilcoxon(d, i, zero_method="wilcox").pvalue
        except ValueError:
            p = float("nan")
        comp_rows.append({
            "Source / Pivot": src, "Metric": m, "Paired N": len(piv),
            "Mean DE direct": d.mean(), "Mean DE indirect": i.mean(),
            "Δ (direct − indirect)": diff.mean(),
            "Direct higher (#sent.)": int((diff > 0).sum()),
            "Equal (#sent.)": int((diff == 0).sum()),
            "Indirect higher (#sent.)": int((diff < 0).sum()),
            "Wilcoxon p": p,
            "Significant (p<.05)": "yes" if p < 0.05 else "no",
        })
comp = pd.DataFrame(comp_rows)

# ---------------------------------------------------------------- 4. Per episode
per_ep = valid.pivot_table(index=["setting", "source", "target"], columns="episode",
                           values=["TED", "ASTrED"], aggfunc="mean")
per_ep.columns = [f"{m} {ep}" for m, ep in per_ep.columns]
per_ep = per_ep.reset_index()
per_ep = per_ep[["setting", "source", "target"]
                + [c for c in per_ep.columns if c.startswith("TED ")]
                + [c for c in per_ep.columns if c.startswith("ASTrED ")]]
per_ep.columns = ["Setting", "Source / Pivot", "German target"] + list(per_ep.columns[3:])

# ---------------------------------------------------------------- 5. Sentence level
sent = df[["setting", "source", "target", "episode", "sent_id", "src_text", "tgt_text",
           "src_len", "tgt_len", "excluded", "TED", "ASTrED", "TED_norm", "ASTrED_norm",
           "n_aligns", "word_cross", "seq_cross", "sacr_cross", "aligns", "align_identical"]]

# ---------------------------------------------------------------- 0. Final summary (presentation table)
NAMES = {"ES source": "Spanish source", "EN pivot": "English pivot", "EN_CC pivot": "English CC pivot"}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


final_main = pd.DataFrame([{
    "Setting": f"({r['Setting']})",
    "Source / pivot": NAMES[r["Source / Pivot"]],
    "German target": r["German target"].replace("DE", "German"),
    "N pairs": r["N pairs"],
    "TED mean": r["TED mean"], "TED SD": r["TED SD"],
    "ASTrED mean": r["ASTrED mean"], "ASTrED SD": r["ASTrED SD"],
} for _, r in summary.iterrows()])

final_comp = []
for src in SRC_ORDER:
    c = comp[comp["Source / Pivot"] == src].set_index("Metric")
    final_comp.append({
        "Source / pivot": NAMES[src], "Paired N": int(c.loc["TED", "Paired N"]),
        "Δ TED (direct − indirect)": c.loc["TED", "Δ (direct − indirect)"],
        "TED sig.": stars(c.loc["TED", "Wilcoxon p"]), "TED p": c.loc["TED", "Wilcoxon p"],
        "Δ ASTrED (direct − indirect)": c.loc["ASTrED", "Δ (direct − indirect)"],
        "ASTrED sig.": stars(c.loc["ASTrED", "Wilcoxon p"]), "ASTrED p": c.loc["ASTrED", "Wilcoxon p"],
    })
final_comp = pd.DataFrame(final_comp)

notes = pd.DataFrame({"Notes": [
    "Data: Club de Cuervos multi-parallel test set, 5 episodes (S01E02, S02E06, S02E08, S02E10, S03E07), 3,276 aligned sentences.",
    "Versions: ES = Spanish source; EN = English pivot (Netflix subtitles); EN_CC = English closed captions pivot;",
    "          DE direct = DE_Direkt (commissioned, translated directly from ES); DE indirect = DE (Netflix, indirect i.e. via an English pivot; which English version is not documented).",
    "Parses: Stanza UD (tokenize, mwt, pos, lemma, depparse) from Subtitles Data/conllu; multi-word token ranges skipped, syntactic words used.",
    "TED: tree edit distance (APTED, unit costs for insert/delete/rename) between source and target dependency trees, nodes labelled by deprel (subtypes stripped).",
    "ASTrED: Aligned Syntactic Tree Edit Distance (Vanroy, De Clercq, Tezcan, Daems & Macken 2021, Computational Linguistics 47(4); 'astred' 0.9.7) – TED over nodes labelled by word-alignment-connected deprel groups.",
    "Word alignment: awesome-align 0.1.7 through astred's Aligner (defaults: mBERT bert-base-multilingual-cased, layer 8, softmax extraction, threshold 0.001, no fine-tuning).",
    "  Compatibility patch: astred 0.9.7 expects an older awesome-align model constructor; only model loading was adapted, alignment code is astred's/awesome-align's own.",
    "  Cross-check: alignments were also computed with an independent re-implementation; agreement is in Sentence_Scores column 'align_identical'.",
    "Punctuation (incl. subtitle dialogue dashes '-') is kept as tree nodes, as produced by the parser.",
    "Normalised scores: score / (number of source words + number of target words), per sentence, then averaged.",
    "Excluded: sentence pairs where either side contains only punctuation (e.g. '-' placeholders for unmatched subtitles).",
    "Direct vs indirect: paired comparison over the same source sentences (pairs valid in both settings); two-sided Wilcoxon signed-rank test.",
    "Higher values = greater syntactic divergence between the source/pivot sentence and the German subtitle.",
]})

# ---------------------------------------------------------------- Write
FC_ROW = len(final_main) + 6  # first row of the comparison table header block
with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    final_main.to_excel(xw, sheet_name="Final_Summary", index=False, startrow=2)
    final_comp.to_excel(xw, sheet_name="Final_Summary", index=False, startrow=FC_ROW)
    summary.to_excel(xw, sheet_name="Summary", index=False, startrow=1)
    matrix.to_excel(xw, sheet_name="Summary", index=False, startrow=len(summary) + 5)
    comp.to_excel(xw, sheet_name="Direct_vs_Indirect", index=False, startrow=1)
    per_ep.to_excel(xw, sheet_name="Per_Episode", index=False, startrow=1)
    sent.to_excel(xw, sheet_name="Sentence_Scores", index=False)
    notes.to_excel(xw, sheet_name="Method", index=False)

# ---------------------------------------------------------------- Styling
wb = load_workbook(OUT)
HEAD = PatternFill("solid", fgColor="1F4E78")
DIRECT = PatternFill("solid", fgColor="E2EFDA")
INDIRECT = PatternFill("solid", fgColor="FCE4D6")
thin = Side(style="thin", color="BFBFBF")
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)


def style_table(ws, header_row, n_rows, n_cols, num_fmt="0.00"):
    for c in range(1, n_cols + 1):
        cell = ws.cell(header_row, c)
        cell.fill, cell.font = HEAD, Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        cell.border = BORDER
    for r in range(header_row + 1, header_row + n_rows + 1):
        for c in range(1, n_cols + 1):
            cell = ws.cell(r, c)
            cell.border = BORDER
            if isinstance(cell.value, float):
                cell.number_format = num_fmt
    ws.row_dimensions[header_row].height = 32


def title(ws, cell, text):
    ws[cell] = text
    ws[cell].font = Font(bold=True, size=13, color="1F4E78")


def autosize(ws, width=None, first=None):
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        ws.column_dimensions[letter].width = width or 14
    if first:
        ws.column_dimensions["A"].width = first


# Final summary
ws = wb["Final_Summary"]
title(ws, "A1", "TED and ASTrED: source/pivot → German subtitles (mean per sentence)")
style_table(ws, 3, len(final_main), final_main.shape[1])
for r in range(4, 4 + len(final_main)):
    fill = INDIRECT if "indirect" in ws.cell(r, 3).value else DIRECT
    for c in range(1, final_main.shape[1] + 1):
        ws.cell(r, c).fill = fill
title(ws, f"A{FC_ROW}", "Direct − indirect German (paired on the same sentences, Wilcoxon signed-rank)")
style_table(ws, FC_ROW + 1, len(final_comp), final_comp.shape[1])
for r in range(FC_ROW + 2, FC_ROW + 2 + len(final_comp)):
    for c in (5, 8):
        ws.cell(r, c).number_format = "0.0000" if ws.cell(r, c).value >= 0.0001 else "0.00E+00"
    for c in (4, 7):
        if ws.cell(r, c).value != "n.s.":
            ws.cell(r, c).font = Font(bold=True, color="C00000")
note_row = FC_ROW + len(final_comp) + 3
for i, txt in enumerate([
    "Higher values = more syntactic divergence between the source/pivot sentence and the German subtitle.",
    "Δ > 0: direct German diverges more from that source/pivot; Δ < 0: indirect German diverges more.",
    "* p<.05, ** p<.01, *** p<.001, n.s. = not significant. Green = direct, orange = indirect. Details in other sheets.",
]):
    ws.cell(note_row + i, 1, txt).font = Font(italic=True, color="595959")
for col, wdt in zip("ABCDEFGH", [16, 18, 22, 22, 14, 26, 14, 14]):
    ws.column_dimensions[col].width = wdt

# Summary
ws = wb["Summary"]
title(ws, "A1", "TED and ASTrED: source/pivot → German subtitles (direct vs. indirect), mean per sentence")
style_table(ws, 2, len(summary), summary.shape[1])
for r in range(3, 3 + len(summary)):
    fill = DIRECT if "direct" == ws.cell(r, 3).value.split()[-1] else INDIRECT
    for c in (1, 2, 3):
        ws.cell(r, c).fill = fill
    for c in (14, 15):
        ws.cell(r, c).number_format = "0"
for col in ("F", "J"):
    ws.conditional_formatting.add(f"{col}3:{col}{2 + len(summary)}",
                                  ColorScaleRule(start_type="min", start_color="F8F8F8",
                                                 end_type="max", end_color="F4B183"))
mrow = len(summary) + 5
title(ws, f"A{mrow}", "Overview matrix (mean per sentence)")
style_table(ws, mrow + 1, len(matrix), matrix.shape[1])
autosize(ws, 13, 9)
ws.column_dimensions["B"].width = 15
ws.column_dimensions["C"].width = 15
ws.freeze_panes = "A3"

# Charts
chart_row = mrow + len(matrix) + 4
for k, (metric, cols) in enumerate([("TED", (2, 3)), ("ASTrED", (4, 5))]):
    ch = BarChart()
    ch.type, ch.grouping = "col", "clustered"
    ch.title = f"Mean {metric} per sentence"
    ch.y_axis.title = metric
    ch.y_axis.majorGridlines = None
    data = Reference(ws, min_col=cols[0], max_col=cols[1], min_row=mrow + 1, max_row=mrow + 1 + len(matrix))
    cats = Reference(ws, min_col=1, min_row=mrow + 2, max_row=mrow + 1 + len(matrix))
    ch.add_data(data, titles_from_data=True)
    ch.set_categories(cats)
    ch.series[0].graphicalProperties.solidFill = "70AD47"
    ch.series[1].graphicalProperties.solidFill = "ED7D31"
    ch.height, ch.width = 7.5, 13
    ws.add_chart(ch, f"{'A' if k == 0 else 'H'}{chart_row}")

# Direct vs indirect
ws = wb["Direct_vs_Indirect"]
title(ws, "A1", "Direct vs. indirect German subtitles – paired comparison per source/pivot (Wilcoxon signed-rank)")
style_table(ws, 2, len(comp), comp.shape[1])
for r in range(3, 3 + len(comp)):
    ws.cell(r, 10).number_format = "0.0000"
    if ws.cell(r, 11).value == "yes":
        ws.cell(r, 11).font = Font(bold=True, color="C00000")
    if ws.cell(r, 2).value.endswith("norm"):
        for c in (4, 5, 6):
            ws.cell(r, c).number_format = "0.000"
autosize(ws, 14, 14)

# Per episode
ws = wb["Per_Episode"]
title(ws, "A1", "Mean TED / ASTrED per episode")
style_table(ws, 2, len(per_ep), per_ep.shape[1])
autosize(ws, 12, 9)
ws.column_dimensions["B"].width = 15
ws.column_dimensions["C"].width = 15

# Sentence scores
ws = wb["Sentence_Scores"]
style_table(ws, 1, len(sent), sent.shape[1], num_fmt="0.000")
autosize(ws, 11)
ws.column_dimensions["F"].width = 60
ws.column_dimensions["G"].width = 60
ws.freeze_panes = "A2"
ws.auto_filter.ref = ws.dimensions

ws = wb["Method"]
ws.column_dimensions["A"].width = 160
ws["A1"].fill, ws["A1"].font = HEAD, Font(bold=True, color="FFFFFF")

FIG_ORDER = ["summary_table", "TED_mean_barplot", "ASTrED_mean_barplot", "TED_paired_difference",
             "ASTrED_paired_difference", "TED_boxplot", "ASTrED_boxplot", "TED_KDE", "ASTrED_KDE",
             "TED_per_episode", "ASTrED_per_episode"]
figs = sorted((HERE / "figures").glob("*.png"),
              key=lambda f: FIG_ORDER.index(f.stem) if f.stem in FIG_ORDER else len(FIG_ORDER))
if figs:
    from openpyxl.drawing.image import Image as XLImage

    ws = wb.create_sheet("Figures", 1)
    title(ws, "A1", "Figures (regenerate with make_figures.py before build_excel.py)")
    row = 3
    for f in figs:
        img = XLImage(str(f))
        scale = 900 / img.width
        img.width, img.height = int(img.width * scale), int(img.height * scale)
        ws.add_image(img, f"A{row}")
        row += int(img.height / 20) + 3

wb.save(OUT)
print(f"saved {OUT}")
print(summary.round(3).to_string(index=False))
print()
print(comp.round(4).to_string(index=False))
