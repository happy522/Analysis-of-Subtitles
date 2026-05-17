from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

files = [
    BASE_DIR / "Subtitles Data/de.csv",
    BASE_DIR / "Subtitles Data/en.csv",
    BASE_DIR / "Subtitles Data/es.csv",
    BASE_DIR / "Subtitles Data/DE_Direkt.csv"

]

dfs = []

for file in files:
    df = pd.read_csv(file,sep = "\t")

    # remove fully empty rows
    df = df.dropna(how="all")

    # set alang value only for DE_Direkt.csv
    if file.name == "DE_Direkt.csv":
        df["alang"] = "DE_Direkt"

    dfs.append(df)

# merge all files
merged_df = pd.concat(dfs, ignore_index=True)

# save merged file
output_path = BASE_DIR / "Subtitles Data/merged.csv"

merged_df.to_csv(output_path, index=False)

print(f"Merged CSV saved to: {output_path}")