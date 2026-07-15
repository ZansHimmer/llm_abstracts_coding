import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

# --- CONFIG ---
ID_COL = "MesH_ID"
LLM_DECISION_COL = "decision_LLM_2"

RUNS = {
    "run_1_QW0": r"matched_sheets\matched_master_sheet_2_qwen3_8b_bs-1.xlsx",
    "run_2_QW1": r"matched_sheets\matched_master_sheet_2-temp1_qwen3_8b_bs-1.xlsx",
}

OUTPUT_DIR = Path(r"record_level_variability\qwen3_8b_bs-1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- HELPERS ---
def read_run(run_name, file_path):
    df = pd.read_excel(file_path, usecols=[ID_COL, LLM_DECISION_COL])
    df = df.rename(columns={LLM_DECISION_COL: run_name})

    if df[ID_COL].duplicated().any():
        duplicated_ids = df.loc[df[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(
            f"Duplicate {ID_COL}s found in {run_name}: {duplicated_ids[:10]}"
        )

    df[run_name] = pd.to_numeric(df[run_name], errors="coerce")
    return df


def fleiss_kappa(ratings_df):
    """
    Fleiss' kappa for any number of runs/raters.
    Rows = records, columns = runs.
    Assumes no missing values.
    """
    ratings = ratings_df.to_numpy()
    categories = sorted(pd.unique(ratings_df.values.ravel()))

    n_records, n_raters = ratings.shape

    counts = np.array([
        [(row == category).sum() for category in categories]
        for row in ratings
    ])

    p_i = ((counts ** 2).sum(axis=1) - n_raters) / (
        n_raters * (n_raters - 1)
    )
    p_bar = p_i.mean()

    p_j = counts.sum(axis=0) / (n_records * n_raters)
    p_e = (p_j ** 2).sum()

    if p_e == 1:
        return np.nan

    return (p_bar - p_e) / (1 - p_e)


# --- LOAD ALL RUNS ---
dfs = [read_run(run_name, path) for run_name, path in RUNS.items()]
run_cols = list(RUNS.keys())

# Outer merge keeps all MesH_IDs, even if missing in some runs
df_all = dfs[0]
for next_df in dfs[1:]:
    df_all = df_all.merge(next_df, on=ID_COL, how="outer")

# --- CHECK NON-OVERLAPPING IDs ---
df_all["missing_in_runs"] = df_all[run_cols].isna().apply(
    lambda row: ", ".join(row.index[row]),
    axis=1
)

df_nonoverlap_ids = df_all[df_all["missing_in_runs"] != ""].copy()

print(f"Total unique {ID_COL}s across all runs: {len(df_all)}")
print(f"{ID_COL}s not present in all runs: {len(df_nonoverlap_ids)}")

df_nonoverlap_ids.to_excel(
    OUTPUT_DIR / "mesh_ids_not_present_in_all_runs.xlsx",
    index=False
)

# --- AGREEMENT ANALYSIS: ONLY RECORDS PRESENT IN ALL RUNS ---
df_eval = df_all.dropna(subset=run_cols).copy()

# Keep only numeric LLM decisions
for col in run_cols:
    df_eval[col] = pd.to_numeric(df_eval[col], errors="coerce")

df_eval = df_eval.dropna(subset=run_cols).copy()

# Only keep binary decisions
binary_mask = df_eval[run_cols].isin([0, 1]).all(axis=1)
df_nonbinary = df_eval[~binary_mask].copy()

df_nonbinary.to_excel(
    OUTPUT_DIR / "mesh_ids_with_nonbinary_decisions.xlsx",
    index=False
)

df_eval = df_eval[binary_mask].copy()
df_eval[run_cols] = df_eval[run_cols].astype(int)

# --- RECORD-LEVEL AGREEMENT / DISAGREEMENT ---
df_eval["n_unique_decisions"] = df_eval[run_cols].nunique(axis=1)
df_eval["classification_changed"] = df_eval["n_unique_decisions"] > 1
df_eval["all_runs_agree"] = df_eval["n_unique_decisions"] == 1

# --- CLASSIFY VOTE PATTERNS ---
n_runs = len(run_cols)

df_eval["n_include_votes"] = df_eval[run_cols].sum(axis=1)
df_eval["n_exclude_votes"] = n_runs - df_eval["n_include_votes"]

df_eval["vote_pattern"] = (
    df_eval["n_include_votes"].astype(str)
    + "_include__"
    + df_eval["n_exclude_votes"].astype(str)
    + "_exclude"
)

df_eval["majority_decision"] = np.where(
    df_eval["n_include_votes"] > df_eval["n_exclude_votes"],
    1,
    np.where(
        df_eval["n_exclude_votes"] > df_eval["n_include_votes"],
        0,
        np.nan
    )
)

df_eval["agreement_type"] = np.select(
    [
        df_eval["n_include_votes"].eq(n_runs),
        df_eval["n_exclude_votes"].eq(n_runs),
        df_eval["n_include_votes"].eq(df_eval["n_exclude_votes"]),
        df_eval["classification_changed"],
    ],
    [
        "unanimous_include",
        "unanimous_exclude",
        "tie_split",
        "majority_disagreement",
    ],
    default="unknown"
)

df_eval["minority_vote_count"] = df_eval[
    ["n_include_votes", "n_exclude_votes"]
].min(axis=1)

df_eval["majority_margin"] = (
    df_eval["n_include_votes"] - df_eval["n_exclude_votes"]
).abs()

# --- OVERALL AGREEMENT METRICS ---
prop_changed = df_eval["classification_changed"].mean()
prop_all_agree = df_eval["all_runs_agree"].mean()

print(f"\nRecords evaluated for agreement: {len(df_eval)}")
print(f"Proportion with changed classification across runs: {prop_changed:.4f}")
print(f"All-run agreement: {prop_all_agree:.4f}")

# --- PAIRWISE AGREEMENT ---
pairwise_agreement_rows = []

print("\nPairwise agreement:")
for a, b in combinations(run_cols, 2):
    n_agree = int((df_eval[a] == df_eval[b]).sum())
    n_disagree = int((df_eval[a] != df_eval[b]).sum())

    agreement = n_agree / len(df_eval)
    disagreement = n_disagree / len(df_eval)

    print(f"{a} vs {b}: {agreement:.4f}")

    pairwise_agreement_rows.append({
        "run_a": a,
        "run_b": b,
        "n_records": len(df_eval),
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "agreement_rate": agreement,
        "disagreement_rate": disagreement,
    })

pairwise_agreement_summary = pd.DataFrame(pairwise_agreement_rows)

# --- FLEISS' KAPPA ---
kappa = fleiss_kappa(df_eval[run_cols])
print(f"\nFleiss' kappa across {len(run_cols)} runs: {kappa:.4f}")

# --- OVERALL SUMMARY ---
overall_summary = pd.DataFrame([{
    "n_runs": len(run_cols),
    "runs": ", ".join(run_cols),
    "n_records_evaluated": len(df_eval),
    "n_records_changed_classification": int(df_eval["classification_changed"].sum()),
    "proportion_changed_classification": prop_changed,
    "n_records_all_runs_agree": int(df_eval["all_runs_agree"].sum()),
    "all_run_agreement_rate": prop_all_agree,
    "fleiss_kappa": kappa,
}])

# --- AGREEMENT / DISAGREEMENT PATTERN SUMMARY ---
pattern_summary = (
    df_eval
    .groupby(["agreement_type", "vote_pattern"])
    .size()
    .reset_index(name="n_records")
)

pattern_summary["proportion"] = pattern_summary["n_records"] / len(df_eval)

print("\nAgreement / disagreement patterns:")
print(pattern_summary.to_string(index=False))

# --- SAVE SUMMARY WORKBOOK ---
with pd.ExcelWriter(OUTPUT_DIR / "agreement_summary.xlsx") as writer:
    overall_summary.to_excel(
        writer,
        sheet_name="overall_summary",
        index=False
    )

    pairwise_agreement_summary.to_excel(
        writer,
        sheet_name="pairwise_agreement",
        index=False
    )

    pattern_summary.to_excel(
        writer,
        sheet_name="vote_patterns",
        index=False
    )

# --- SAVE RECORDS WHOSE CLASSIFICATION CHANGED ---
df_changed = df_eval[df_eval["classification_changed"]].copy()

df_changed.to_excel(
    OUTPUT_DIR / "mesh_ids_with_changed_classification_across_runs.xlsx",
    index=False
)

# --- SAVE FULL RECORD-LEVEL AGREEMENT OUTPUT ---
df_eval.to_excel(
    OUTPUT_DIR / "record_level_agreement_across_runs.xlsx",
    index=False
)

print("\nDone.")
print(f"Outputs saved in: {OUTPUT_DIR}")