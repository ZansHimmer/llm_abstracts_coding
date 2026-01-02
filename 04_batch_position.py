import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

file_path = 'shuffled_master_sheet.xlsx'

df = pd.read_excel(file_path)
df['MesH_ID'] = df['MesH_ID'].astype(str).str.strip()

decisions = []

for f in Path("outputs_2_gpt-5-mini_bs-20").glob("*_coding_output_2.txt"):
    tmp = pd.read_csv(
        f,
        header=None,
        names=["MesH_ID", "decision"],
    )

    tmp['MesH_ID'] = tmp['MesH_ID'].astype(str).str.strip()

    tmp['position_in_file'] = tmp.index + 1

    tmp['source_file'] = f.name

    decisions.append(tmp)

decisions_df = pd.concat(decisions, ignore_index=True)

merged_df = df.merge(decisions_df, on="MesH_ID", how="left")

merged_df = merged_df.rename(columns={
    "decision": "decision_LLM_2"
})

print(merged_df.head())

output_path = "matched_sheets\\matched_master_sheet_2_gpt-5-mini_bs-20_with_positions.xlsx"
merged_df.to_excel(output_path, index=False)


merged_df['final-decision_include'] = pd.to_numeric(merged_df['final-decision_include'], errors='coerce')
merged_df['decision_LLM_2'] = pd.to_numeric(merged_df['decision_LLM_2'], errors='coerce')
df_eval = merged_df.dropna(subset=['final-decision_include', 'decision_LLM_2'])

results = []
report_rows = []

for pos in sorted(df_eval['position_in_file'].dropna().unique()):
    df_pos = df_eval[df_eval['position_in_file'] == pos].copy()

    y_true = df_pos['final-decision_include'].astype(int)
    y_pred = df_pos['decision_LLM_2'].astype(int)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    results.append({
        'position_in_file': pos,
        'n_samples': len(df_pos),
        'accuracy': acc,
        'tn': cm[0, 0],
        'fp': cm[0, 1],
        'fn': cm[1, 0],
        'tp': cm[1, 1],
    })

    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    for label, metrics in report.items():
        if label in ['accuracy', 'macro avg', 'weighted avg']:
            continue

        report_rows.append({
            'position_in_file': pos,
            'label': label,
            **metrics})


results_df = pd.DataFrame(results)
print(results_df)

report_df = pd.DataFrame(report_rows)
print(report_df)

with pd.ExcelWriter("evaluations\\evaluation_by_position_2_gpt-5-mini_bs-20.xlsx", engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="Results", index=False)
    report_df.to_excel(writer, sheet_name="Detailed Report", index=False)