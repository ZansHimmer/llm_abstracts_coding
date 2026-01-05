import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = 'shuffled_master_sheet.xlsx'

df = pd.read_excel(file_path)
df['MesH_ID'] = df['MesH_ID'].astype(str).str.strip()

decisions = []

for f in Path("outputs_5_gpt-5-mini_bs-5").glob("*.txt"):
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

output_path = (
    "matched_sheets\\matched_master_sheet_5_gpt-5-mini_bs-5_with_positions.xlsx"
)
merged_df.to_excel(output_path, index=False)

merged_df['final-decision_include'] = pd.to_numeric(
    merged_df['final-decision_include'], errors='coerce'
)
merged_df['decision_LLM_2'] = pd.to_numeric(
    merged_df['decision_LLM_2'], errors='coerce'
)

df_eval = merged_df.dropna(
    subset=['final-decision_include', 'decision_LLM_2']
)

results = []
report_rows = []

for pos in sorted(df_eval['position_in_file'].dropna().unique()):
    df_pos = df_eval[df_eval['position_in_file'] == pos]

    y_true = df_pos['final-decision_include'].astype(int)
    y_pred = df_pos['decision_LLM_2'].astype(int)

    cm = confusion_matrix(y_true, y_pred)

    results.append({
        'position_in_file': pos,
        'n_samples': len(df_pos),
        'accuracy': (cm[0, 0] + cm[1, 1]) / cm.sum(),
        'tn': cm[0, 0],
        'fp': cm[0, 1],
        'fn': cm[1, 0],
        'tp': cm[1, 1],
    })

    report = classification_report(
        y_true, y_pred, digits=4, output_dict=True
    )

    for label, metrics in report.items():
        if label in ['accuracy', 'macro avg', 'weighted avg']:
            continue

        report_rows.append({
            'position_in_file': pos,
            'label': label,
            **metrics
        })

results_df = pd.DataFrame(results)
report_df = pd.DataFrame(report_rows)

def position_region(position, batch_size):
    if position <= 0.25 * batch_size:
        return "beginning"
    elif position <= 0.75 * batch_size:
        return "middle"
    else:
        return "end"

batch_sizes = (
    df_eval
    .groupby("source_file")["position_in_file"]
    .max()
    .rename("batch_size")
    .reset_index()
)

df_eval = df_eval.merge(
    batch_sizes,
    on="source_file",
    how="left"
)

df_eval["position_region"] = df_eval.apply(
    lambda r: position_region(
        r["position_in_file"],
        r["batch_size"]
    ),
    axis=1
)

region_results = []

for region in ["beginning", "middle", "end"]:
    df_r = df_eval[df_eval["position_region"] == region]

    y_true = df_r["final-decision_include"].astype(int)
    y_pred = df_r["decision_LLM_2"].astype(int)

    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    region_results.append({
        "position_region": region,
        "n_samples": len(df_r),
        "accuracy": report["accuracy"],
        "precision_0": report["0"]["precision"],
        "recall_0": report["0"]["recall"],
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
    })

region_results_df = pd.DataFrame(region_results)

with pd.ExcelWriter(
    "evaluations\\evaluation_by_position_5_gpt-5-mini_bs-5.xlsx",
    engine="openpyxl"
) as writer:
    results_df.to_excel(writer, sheet_name="Results", index=False)
    report_df.to_excel(writer, sheet_name="Detailed Report", index=False)
    region_results_df.to_excel(
        writer,
        sheet_name="Position_Regions",
        index=False
    )
