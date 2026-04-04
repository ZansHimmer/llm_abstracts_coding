import pandas as pd
from pathlib import Path

# version 04/03/2026

from master_config import SETTINGS, MASTER_SHEET_SHUFFLED, MASTER_SHEET_FULL



file_path = MASTER_SHEET_SHUFFLED

df = pd.read_excel(file_path)
df['MesH_ID'] = df['MesH_ID'].astype(str).str.strip()

decisions = []

for f in Path(SETTINGS.outputs_dir()).glob("*.txt"):
    try:
        tmp = pd.read_csv(f, header=None)

        # skip empty files
        if tmp.shape[1] < 2:
            continue

        # keep only first two columns
        tmp = tmp.iloc[:, :2]
        tmp.columns = ["MesH_ID", "decision"]

        tmp['MesH_ID'] = tmp['MesH_ID'].astype(str).str.strip()
        decisions.append(tmp)

    except Exception:
        continue

# only concat if we actually have data
if decisions:
    decisions_df = pd.concat(decisions, ignore_index=True)
else:
    decisions_df = pd.DataFrame(columns=["MesH_ID", "decision"])

merged_df = df.merge(decisions_df, on="MesH_ID", how="left")
merged_df = merged_df.rename(columns={"decision": "decision_LLM_2"})

print(merged_df.head())

output_path = SETTINGS.matched_sheet_path(full=False)
output_path.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_excel(output_path, index=False)