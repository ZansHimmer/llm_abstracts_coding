import pandas as pd
from pathlib import Path

file_path = 'master-sheet-full.xlsx'

df = pd.read_excel(file_path)
df['MesH_ID'] = df['MesH_ID'].astype(str).str.strip()

decisions = []

for f in Path("outputs_8_gpt-5-mini_bs-1").glob("*.txt"):
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

output_path = "matched_sheets\\matched_master_sheet_full_8_gpt-5-mini_bs-1.xlsx"
merged_df.to_excel(output_path, index=False)
