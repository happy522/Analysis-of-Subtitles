"""Build TED_ASTrED/TED_ASTrED_results.xlsx (Summary, Direct_vs_Indirect, Per_Episode) from sentence_scores.csv."""
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
OUT = HERE / "TED_ASTrED_results.xlsx"

df = pd.read_csv(HERE / "sentence_scores.csv")
v = df[~df["excluded"]]

SRC = ["ES source", "EN pivot", "EN_CC pivot"]
NAMES = {"ES source": "Spanish source", "EN pivot": "English pivot", "EN_CC pivot": "English CC pivot"}
EPS = ["S01E02", "S02E06", "S02E08", "S02E10", "S03E07"]

# Summary: one row per setting
rows = []
for sid, g in v.groupby("setting"):
    rows.append({
        "Setting": sid,
        "Source / pivot": NAMES[g["source"].iloc[0]],
        "German target": g["target"].iloc[0].replace("DE", "German"),
        "N pairs": len(g),
        "Excluded pairs": int(df[(df.setting == sid) & df.excluded].shape[0]),
        "TED mean": g.TED.mean(), "TED SD": g.TED.std(), "TED median": g.TED.median(),
        "ASTrED mean": g.ASTrED.mean(), "ASTrED SD": g.ASTrED.std(), "ASTrED median": g.ASTrED.median(),
    })
summary = pd.DataFrame(rows)

# Direct vs indirect: paired on the same sentences
rows = []
for s in SRC:
    piv = v[v.source == s].pivot_table(index=["episode", "sent_id"], columns="direction",
                                       values=["TED", "ASTrED"]).dropna()
    for m in ["TED", "ASTrED"]:
        d, i = piv[(m, "direct")], piv[(m, "indirect")]
        rows.append({
            "Source / pivot": NAMES[s], "Metric": m, "Paired N": len(piv),
            "Mean German direct": d.mean(), "Mean German indirect": i.mean(),
            "Difference (direct - indirect)": (d - i).mean(),
            "Wilcoxon p": wilcoxon(d, i).pvalue,
        })
comp = pd.DataFrame(rows)

# Per episode
per_ep = v.pivot_table(index=["setting", "source", "target"], columns="episode", values=["TED", "ASTrED"])
per_ep = per_ep[[(m, e) for m in ["TED", "ASTrED"] for e in EPS]]
per_ep.columns = [f"{m} {e}" for m, e in per_ep.columns]
per_ep = per_ep.reset_index()
per_ep["source"] = per_ep["source"].map(NAMES)
per_ep["target"] = per_ep["target"].str.replace("DE", "German")
per_ep = per_ep.rename(columns={"setting": "Setting", "source": "Source / pivot", "target": "German target"})

with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    summary.to_excel(xw, sheet_name="Summary", index=False)
    comp.to_excel(xw, sheet_name="Direct_vs_Indirect", index=False)
    per_ep.to_excel(xw, sheet_name="Per_Episode", index=False)

# Light formatting: bold header, 2 decimals, column widths, frozen header
wb = load_workbook(OUT)
for ws in wb.worksheets:
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if isinstance(cell.value, float):
                header = ws.cell(1, cell.column).value
                cell.number_format = "0.0000" if header == "Wilcoxon p" and cell.value >= 1e-4 else \
                    "0.00E+00" if header == "Wilcoxon p" else "0.00"
    for col in ws.columns:
        width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max(width, 8) + 2, 32)
    ws.freeze_panes = "A2"
wb.save(OUT)
print(f"saved {OUT}")
