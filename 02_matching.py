import pandas as pd
from pathlib import Path

file_path = 'shuffled_master_sheet.xlsx'

df = pd.read_excel(file_path)
df['MesH_ID'] = df['MesH_ID'].astype(str).str.strip()

decisions = []

for f in Path("outputs_2_qwen3_8b_bs-1").glob("*.txt"):
    tmp = pd.read_csv(
        f,
        header=None,
        names=["MesH_ID", "decision"],
    )
    tmp['MesH_ID'] = tmp['MesH_ID'].astype(str).str.strip()
    decisions.append(tmp)

decisions_df = pd.concat(decisions, ignore_index=True) 

merged_df = df.merge(decisions_df, on="MesH_ID", how="left")
merged_df = merged_df.rename(columns={"decision": "decision_LLM_2"})

print(merged_df.head())

output_path = "matched_sheets\\matched_master_sheet_2_qwen3_8b_bs-1.xlsx"
merged_df.to_excel(output_path, index=False)
