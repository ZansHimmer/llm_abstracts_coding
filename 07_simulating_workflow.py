import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIG ---
ID_COL = "MesH_ID"
FINAL_HUMAN_COL = "final-decision_include"
LLM_DECISION_COL = "decision_LLM_2"

OUTPUT_DIR = Path(r"subsample_screening_performance\qwen3_8b_bs-1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Start with one matched sheet.
# Later, add more sheets here.
MATCHED_SHEETS = {
    "run_1": r"matched_sheets\matched_master_sheet_2_gpt-4.1-mini_bs-1.xlsx",
    # "run_2": r"matched_sheets\another_matched_sheet.xlsx",
    # "run_3": r"matched_sheets\another_matched_sheet.xlsx",
}

# 10%, 15%, 20% samples
SAMPLE_PROPORTIONS = [0.10, 0.15, 0.20]

N_DRAWS = 1000
RANDOM_SEED = 123

# If True, each subsample preserves the human include/exclude distribution.
# False is recommended for testing the propsed workflow
STRATIFY_BY_HUMAN_LABEL = False


# --- HELPERS ---
def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def compute_metrics(df):
    """
    Computes screening performance for binary labels:
    final human decision = ground truth
    LLM decision = prediction
    """
    y_true = df[FINAL_HUMAN_COL].astype(int)
    y_pred = df[LLM_DECISION_COL].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    n = len(df)

    accuracy = safe_divide(tp + tn, n)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)

    if pd.isna(precision) or pd.isna(recall) or (precision + recall == 0):
        f1 = np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # Cohen's kappa
    observed_agreement = accuracy

    p_true_include = safe_divide(tp + fn, n)
    p_true_exclude = safe_divide(tn + fp, n)
    p_pred_include = safe_divide(tp + fp, n)
    p_pred_exclude = safe_divide(tn + fn, n)

    expected_agreement = (
        p_true_include * p_pred_include
        + p_true_exclude * p_pred_exclude
    )

    if expected_agreement == 1:
        kappa = np.nan
    else:
        kappa = safe_divide(
            observed_agreement - expected_agreement,
            1 - expected_agreement
        )

    return {
        "n_records": n,
        "n_human_include": int((y_true == 1).sum()),
        "n_human_exclude": int((y_true == 0).sum()),
        "n_llm_include": int((y_pred == 1).sum()),
        "n_llm_exclude": int((y_pred == 0).sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "kappa": kappa,
    }


def read_matched_sheet(file_path):
    df = pd.read_excel(
        file_path,
        usecols=[ID_COL, FINAL_HUMAN_COL, LLM_DECISION_COL]
    )

    if df[ID_COL].duplicated().any():
        duplicated_ids = df.loc[df[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(f"Duplicate {ID_COL}s found: {duplicated_ids[:10]}")

    df[FINAL_HUMAN_COL] = pd.to_numeric(df[FINAL_HUMAN_COL], errors="coerce")
    df[LLM_DECISION_COL] = pd.to_numeric(df[LLM_DECISION_COL], errors="coerce")

    # Only keep rows with numeric human and LLM decisions
    df = df.dropna(subset=[FINAL_HUMAN_COL, LLM_DECISION_COL]).copy()

    df[[FINAL_HUMAN_COL, LLM_DECISION_COL]] = df[
        [FINAL_HUMAN_COL, LLM_DECISION_COL]
    ].astype(int)

    # Only keep binary labels
    df = df[
        df[FINAL_HUMAN_COL].isin([0, 1])
        & df[LLM_DECISION_COL].isin([0, 1])
    ].copy()

    return df


def draw_sample(df, sample_proportion, random_state):
    if STRATIFY_BY_HUMAN_LABEL:
        sample = (
            df
            .groupby(FINAL_HUMAN_COL, group_keys=False)
            .sample(frac=sample_proportion, random_state=random_state)
        )
    else:
        sample = df.sample(frac=sample_proportion, random_state=random_state)

    remainder = df.drop(index=sample.index)

    return sample, remainder


def add_metric_prefix(metrics, prefix):
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
    }


def summarize_simulations(sim_df):
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "kappa",
    ]

    rows = []

    for (run, sample_proportion), group in sim_df.groupby(
        ["run", "sample_proportion"]
    ):
        for metric in metric_names:
            sample_col = f"sample_{metric}"
            remainder_col = f"remainder_{metric}"
            full_col = f"full_{metric}"

            diff_vs_remainder = group[sample_col] - group[remainder_col]
            abs_diff_vs_remainder = diff_vs_remainder.abs()

            diff_vs_full = group[sample_col] - group[full_col]
            abs_diff_vs_full = diff_vs_full.abs()

            rows.append({
                "run": run,
                "sample_proportion": sample_proportion,
                "metric": metric,

                "full_metric": group[full_col].iloc[0],

                "mean_sample_metric": group[sample_col].mean(),
                "sd_sample_metric": group[sample_col].std(),

                "mean_remainder_metric": group[remainder_col].mean(),
                "sd_remainder_metric": group[remainder_col].std(),

                "mean_difference_sample_minus_remainder": diff_vs_remainder.mean(),
                "mean_abs_difference_sample_vs_remainder": abs_diff_vs_remainder.mean(),
                "p95_abs_difference_sample_vs_remainder": abs_diff_vs_remainder.quantile(0.95),

                "difference_vs_remainder_q025": diff_vs_remainder.quantile(0.025),
                "difference_vs_remainder_q975": diff_vs_remainder.quantile(0.975),

                "mean_abs_difference_sample_vs_full": abs_diff_vs_full.mean(),
                "p95_abs_difference_sample_vs_full": abs_diff_vs_full.quantile(0.95),
            })

    return pd.DataFrame(rows)


# --- RUN ANALYSIS ---
all_full_metrics = []
all_simulation_draws = []

rng = np.random.default_rng(RANDOM_SEED)

for run_name, matched_sheet_path in MATCHED_SHEETS.items():
    print(f"\nAnalyzing {run_name}...")

    df = read_matched_sheet(matched_sheet_path)

    print(f"Records available for analysis: {len(df)}")

    full_metrics = compute_metrics(df)
    full_metrics["run"] = run_name
    all_full_metrics.append(full_metrics)

    for sample_proportion in SAMPLE_PROPORTIONS:
        print(f"  Simulating sample proportion: {sample_proportion}")

        for draw in range(1, N_DRAWS + 1):
            random_state = int(rng.integers(0, 2**32 - 1))

            sample, remainder = draw_sample(
                df=df,
                sample_proportion=sample_proportion,
                random_state=random_state
            )

            sample_metrics = compute_metrics(sample)
            remainder_metrics = compute_metrics(remainder)

            row = {
                "run": run_name,
                "draw": draw,
                "sample_proportion": sample_proportion,
                "n_total": len(df),
                "n_sample": len(sample),
                "n_remainder": len(remainder),
                "stratified_by_human_label": STRATIFY_BY_HUMAN_LABEL,
            }

            row.update(add_metric_prefix(full_metrics, "full"))
            row.update(add_metric_prefix(sample_metrics, "sample"))
            row.update(add_metric_prefix(remainder_metrics, "remainder"))

            all_simulation_draws.append(row)

# --- SAVE OUTPUTS ---
full_metrics_df = pd.DataFrame(all_full_metrics)

simulation_draws_df = pd.DataFrame(all_simulation_draws)

simulation_summary_df = summarize_simulations(simulation_draws_df)

full_metrics_df.to_excel(
    OUTPUT_DIR / "full_dataset_screening_metrics.xlsx",
    index=False
)

simulation_draws_df.to_excel(
    OUTPUT_DIR / "subsample_simulation_draws.xlsx",
    index=False
)

simulation_summary_df.to_excel(
    OUTPUT_DIR / "subsample_simulation_summary.xlsx",
    index=False
)

print("\nDone.")
print(f"Outputs saved in: {OUTPUT_DIR}")