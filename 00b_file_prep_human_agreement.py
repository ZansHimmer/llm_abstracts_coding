import pandas as pd
from pathlib import Path

# --- CONFIG ---
MASTER_SHEET_PATH = r"master-sheet.xlsx"   # change this
OUTPUT_FILE = "human_inclusion_decisions_only.xlsx"

ID_COL = "MesH_ID"
HUMAN_COLS = ["inclusion_1", "inclusion_2"]


# --- LOAD ONLY RELEVANT COLUMNS ---
df = pd.read_excel(
    MASTER_SHEET_PATH,
    usecols=[ID_COL] + HUMAN_COLS
)

# --- CHECK DUPLICATE IDs ---
if df[ID_COL].duplicated().any():
    duplicated_ids = df.loc[df[ID_COL].duplicated(), ID_COL].tolist()
    raise ValueError(f"Duplicate {ID_COL}s found: {duplicated_ids[:10]}")

# --- CONVERT HUMAN DECISIONS TO NUMERIC ---
for col in HUMAN_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --- KEEP ONLY ROWS WITH NUMERIC HUMAN DECISIONS ---
df_clean = df.dropna(subset=HUMAN_COLS).copy()

df_clean[HUMAN_COLS] = df_clean[HUMAN_COLS].astype(int)

# --- SAVE LIGHTWEIGHT FILE ---
df_clean.to_excel(OUTPUT_FILE, index=False)

print(f"Original rows: {len(df)}")
print(f"Rows with numeric inclusion_1 and inclusion_2: {len(df_clean)}")
print(f"Saved to: {OUTPUT_FILE}")