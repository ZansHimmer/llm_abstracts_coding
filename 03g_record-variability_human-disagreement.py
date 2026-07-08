import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIG ---
ID_COL = "MesH_ID"

HUMAN_DECISIONS_FILE = "human_inclusion_decisions_only.xlsx"

HUMAN_COL_1 = "inclusion_1"
HUMAN_COL_2 = "inclusion_2"

LLM_DECISION_COL = "decision_LLM_2"

OUTPUT_DIR = Path(r"human-disagreement_LLM-run-variability\gpt-4.1-mini_bs-1_temp1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add as many runs as you want here.
RUNS = {
    "run_1": r"matched_sheets\matched_master_sheet_2b-temp1_gpt-4.1-mini_bs-1.xlsx",
    "run_2": r"matched_sheets\matched_master_sheet_2c-temp1_gpt-4.1-mini_bs-1.xlsx",
    "run_3": r"matched_sheets\matched_master_sheet_2-temp1_gpt-4.1-mini_bs-1.xlsx",
    # "run_4": r"matched_sheets\matched_master_sheet_2_gpt-4.1-mini_bs-1.xlsx",
}


# --- HELPERS ---
def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def read_human_decisions():
    df = pd.read_excel(
        HUMAN_DECISIONS_FILE,
        usecols=[ID_COL, HUMAN_COL_1, HUMAN_COL_2]
    )

    if df[ID_COL].duplicated().any():
        duplicated_ids = df.loc[df[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(
            f"Duplicate {ID_COL}s found in human decisions file: {duplicated_ids[:10]}"
        )

    df[HUMAN_COL_1] = pd.to_numeric(df[HUMAN_COL_1], errors="coerce")
    df[HUMAN_COL_2] = pd.to_numeric(df[HUMAN_COL_2], errors="coerce")

    # Only keep rows where both original human decisions are numeric
    df = df.dropna(subset=[HUMAN_COL_1, HUMAN_COL_2]).copy()

    df[HUMAN_COL_1] = df[HUMAN_COL_1].astype(int)
    df[HUMAN_COL_2] = df[HUMAN_COL_2].astype(int)

    df["human_disagreement"] = df[HUMAN_COL_1] != df[HUMAN_COL_2]

    df["human_disagreement_label"] = df["human_disagreement"].map({
        False: "no_human_disagreement",
        True: "human_disagreement"
    })

    return df


def read_llm_run(run_name, file_path):
    df = pd.read_excel(
        file_path,
        usecols=[ID_COL, LLM_DECISION_COL]
    )

    if df[ID_COL].duplicated().any():
        duplicated_ids = df.loc[df[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(
            f"Duplicate {ID_COL}s found in {run_name}: {duplicated_ids[:10]}"
        )

    df = df.rename(columns={LLM_DECISION_COL: run_name})
    df[run_name] = pd.to_numeric(df[run_name], errors="coerce")

    return df


# --- LOAD HUMAN DECISIONS ---
human_df = read_human_decisions()

# --- LOAD ALL LLM RUNS ---
run_dfs = [
    read_llm_run(run_name, file_path)
    for run_name, file_path in RUNS.items()
]

run_cols = list(RUNS.keys())
n_runs = len(run_cols)

# Outer merge keeps records even if missing from some runs
llm_all = run_dfs[0]

for next_df in run_dfs[1:]:
    llm_all = llm_all.merge(next_df, on=ID_COL, how="outer")

# --- MERGE HUMAN DISAGREEMENT INFO ONTO LLM RUN DATA ---
df_all = llm_all.merge(human_df, on=ID_COL, how="inner")

# --- CHECK RECORDS MISSING LLM DECISIONS IN AT LEAST ONE RUN ---
df_all["missing_llm_runs"] = df_all[run_cols].isna().apply(
    lambda row: ", ".join(row.index[row]),
    axis=1
)

df_missing_llm = df_all[df_all["missing_llm_runs"] != ""].copy()

df_missing_llm.to_excel(
    OUTPUT_DIR / "records_missing_llm_decision_in_at_least_one_run.xlsx",
    index=False
)

# --- ANALYSIS DATASET ---
# Only use records with:
# - numeric inclusion_1 and inclusion_2
# - LLM decision in all runs
df_eval = df_all.dropna(subset=run_cols).copy()

df_eval[run_cols] = df_eval[run_cols].astype(int)

# Optional but useful: keep only binary LLM decisions
valid_binary_mask = df_eval[run_cols].isin([0, 1]).all(axis=1)
df_non_binary = df_eval[~valid_binary_mask].copy()

df_non_binary.to_excel(
    OUTPUT_DIR / "records_with_non_binary_llm_decisions.xlsx",
    index=False
)

df_eval = df_eval[valid_binary_mask].copy()

# --- CLASSIFY LLM RUN VARIABILITY ---
df_eval["n_unique_llm_decisions"] = df_eval[run_cols].nunique(axis=1)
df_eval["classification_changed_across_runs"] = df_eval["n_unique_llm_decisions"] > 1
df_eval["stable_across_runs"] = df_eval["n_unique_llm_decisions"] == 1

# Since decisions are binary, count include/exclude votes
df_eval["n_include_votes"] = df_eval[run_cols].sum(axis=1)
df_eval["n_exclude_votes"] = n_runs - df_eval["n_include_votes"]

df_eval["vote_pattern"] = (
    df_eval["n_include_votes"].astype(str)
    + "_include__"
    + df_eval["n_exclude_votes"].astype(str)
    + "_exclude"
)

df_eval["llm_variability_label"] = df_eval[
    "classification_changed_across_runs"
].map({
    False: "stable_across_runs",
    True: "changed_across_runs"
})

# --- MAIN SUMMARY: DOES HUMAN DISAGREEMENT PREDICT LLM RUN VARIABILITY? ---
summary = (
    df_eval
    .groupby("human_disagreement_label")
    .agg(
        n_records=(ID_COL, "count"),
        n_changed_across_runs=("classification_changed_across_runs", "sum"),
        proportion_changed_across_runs=("classification_changed_across_runs", "mean"),
        n_stable_across_runs=("stable_across_runs", "sum"),
    )
    .reset_index()
    .rename(columns={"human_disagreement_label": "human_disagreement"})
)

summary.to_excel(
    OUTPUT_DIR / "summary_human_disagreement_vs_llm_run_variability.xlsx",
    index=False
)

print("\nSummary: human disagreement vs LLM run variability")
print(summary.to_string(index=False))

# --- CLEAR CONTINGENCY TABLE ---
contingency = pd.crosstab(
    df_eval["human_disagreement"],
    df_eval["classification_changed_across_runs"],
    rownames=["human_disagreement"],
    colnames=["llm_run_variability"]
)

contingency = contingency.reindex(
    index=[False, True],
    columns=[False, True],
    fill_value=0
)

contingency = contingency.rename(
    index={
        False: "no_human_disagreement",
        True: "human_disagreement",
    },
    columns={
        False: "stable_across_runs",
        True: "changed_across_runs",
    }
)

contingency_export = contingency.reset_index()

contingency_export.to_excel(
    OUTPUT_DIR / "contingency_human_disagreement_vs_llm_run_variability.xlsx",
    index=False
)

print("\nContingency table:")
print(contingency.to_string())

# --- EFFECT MEASURES ---
no_disagreement = df_eval[df_eval["human_disagreement"] == False]
disagreement = df_eval[df_eval["human_disagreement"] == True]

change_rate_no_disagreement = no_disagreement[
    "classification_changed_across_runs"
].mean()

change_rate_disagreement = disagreement[
    "classification_changed_across_runs"
].mean()

risk_difference = change_rate_disagreement - change_rate_no_disagreement
risk_ratio = safe_divide(change_rate_disagreement, change_rate_no_disagreement)

# Odds ratio with 0.5 correction
a = contingency.loc["human_disagreement", "changed_across_runs"]
b = contingency.loc["human_disagreement", "stable_across_runs"]
c = contingency.loc["no_human_disagreement", "changed_across_runs"]
d = contingency.loc["no_human_disagreement", "stable_across_runs"]

odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))

effect_summary = pd.DataFrame([{
    "n_runs": n_runs,
    "n_records_analyzed": len(df_eval),
    "n_human_disagreement": int(df_eval["human_disagreement"].sum()),
    "n_no_human_disagreement": int((~df_eval["human_disagreement"]).sum()),
    "overall_proportion_changed_across_runs": df_eval[
        "classification_changed_across_runs"
    ].mean(),
    "proportion_changed_human_disagreement": change_rate_disagreement,
    "proportion_changed_no_human_disagreement": change_rate_no_disagreement,
    "risk_difference": risk_difference,
    "risk_ratio": risk_ratio,
    "odds_ratio_with_0_5_correction": odds_ratio,
}])

effect_summary.to_excel(
    OUTPUT_DIR / "effect_summary_human_disagreement_vs_llm_run_variability.xlsx",
    index=False
)

print("\nEffect summary:")
print(effect_summary.to_string(index=False))

# --- VOTE PATTERNS BY HUMAN DISAGREEMENT STATUS ---
pattern_summary = (
    df_eval
    .groupby(["human_disagreement_label", "vote_pattern"])
    .size()
    .reset_index(name="n_records")
    .rename(columns={"human_disagreement_label": "human_disagreement"})
)

group_totals = (
    df_eval
    .groupby("human_disagreement_label")
    .size()
    .reset_index(name="n_total_records")
    .rename(columns={"human_disagreement_label": "human_disagreement"})
)

pattern_summary = pattern_summary.merge(
    group_totals,
    on="human_disagreement",
    how="left"
)

pattern_summary["proportion_within_human_group"] = (
    pattern_summary["n_records"] / pattern_summary["n_total_records"]
)

pattern_summary.to_excel(
    OUTPUT_DIR / "vote_pattern_summary_by_human_disagreement.xlsx",
    index=False
)

# --- RECORD-LEVEL OUTPUTS ---
df_eval.to_excel(
    OUTPUT_DIR / "record_level_human_disagreement_and_llm_run_variability.xlsx",
    index=False
)

df_changed = df_eval[df_eval["classification_changed_across_runs"]].copy()

df_changed.to_excel(
    OUTPUT_DIR / "records_with_changed_llm_classification_across_runs.xlsx",
    index=False
)

print("\nDone.")
print(f"Outputs saved in: {OUTPUT_DIR}")