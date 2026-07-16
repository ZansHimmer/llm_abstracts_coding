import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

# --- CONFIG ---
ID_COL = "MesH_ID"
FINAL_HUMAN_COL = "final-decision_include"
LLM_DECISION_COL = "decision_LLM_2"

OUTPUT_DIR = Path("paired_bootstrap_all_pairwise_comparisons/reasoning_batch_comparisons_medium")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHED_SHEETS = {
    "gpt-5-mini_medium-reasoning_bs-1": r"matched_sheets\matched_master_sheet_2_gpt-5-mini_bs-1.xlsx",
    "gpt-5-mini_medium-reasoning_bs-5": r"matched_sheets\matched_master_sheet_2_gpt-5-mini_bs-5.xlsx",
    "gpt-5-mini_medium-reasoning_bs-10": r"matched_sheets\matched_master_sheet_2_gpt-5-mini_bs-10.xlsx",
    "gpt-5-mini_medium-reasoning_bs-20": r"matched_sheets\matched_master_sheet_2_gpt-5-mini_bs-20.xlsx",
    "gpt-5-mini_medium-reasoning_bs-100": r"matched_sheets\matched_master_sheet_2_gpt-5-mini_bs-100.xlsx",
}

CONFIDENCE = 0.95
N_BOOTSTRAP = 5000
RANDOM_SEED = 123

SAVE_BOOTSTRAP_DRAWS = False


# --- HELPERS ---
def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def cohen_kappa_binary(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)

    if n == 0:
        return np.nan

    observed_agreement = np.mean(y_true == y_pred)

    p_true_1 = np.mean(y_true == 1)
    p_true_0 = np.mean(y_true == 0)

    p_pred_1 = np.mean(y_pred == 1)
    p_pred_0 = np.mean(y_pred == 0)

    expected_agreement = (p_true_1 * p_pred_1) + (p_true_0 * p_pred_0)

    if expected_agreement == 1:
        return np.nan

    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    n = tp + tn + fp + fn

    return {
        "accuracy": safe_divide(tp + tn, n),
        "recall": safe_divide(tp, tp + fn),
        "precision": safe_divide(tp, tp + fp),
        "cohen_kappa": cohen_kappa_binary(y_true, y_pred),
        "n_records": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_human_include": tp + fn,
        "n_human_exclude": tn + fp,
        "n_llm_include": tp + fp,
        "n_llm_exclude": tn + fn,
        "base_rate": safe_divide(tp + fn, n),
    }


def read_run(run_name, file_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    required_cols = [ID_COL, FINAL_HUMAN_COL, LLM_DECISION_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"{run_name} is missing columns: {missing_cols}\n"
            f"Columns found: {list(df.columns)}"
        )

    df = df[required_cols].copy()

    if df[ID_COL].duplicated().any():
        duplicate_rows = df[df[ID_COL].duplicated(keep=False)].copy()

        duplicate_rows.to_excel(
            OUTPUT_DIR / f"duplicate_rows_dropped_first_occurrence__{run_name}.xlsx",
            index=False
        )

        n_before = len(df)

        # Drop the first occurrence of each duplicated MesH_ID.
        # Keep the last occurrence.
        df = df.drop_duplicates(
            subset=[ID_COL],
            keep="last"
        ).copy()

        n_after = len(df)

        print(
            f"{run_name}: dropped {n_before - n_after} duplicate rows "
            f"by keeping the last occurrence of each {ID_COL}."
        )

    df = df.rename(
        columns={
            FINAL_HUMAN_COL: f"human_final__{run_name}",
            LLM_DECISION_COL: f"llm_decision__{run_name}",
        }
    )

    df[f"human_final__{run_name}"] = pd.to_numeric(
        df[f"human_final__{run_name}"],
        errors="coerce"
    )

    df[f"llm_decision__{run_name}"] = pd.to_numeric(
        df[f"llm_decision__{run_name}"],
        errors="coerce"
    )

    return df


def prepare_pair_data(run_a, run_b, run_dfs):
    df = run_dfs[run_a].merge(
        run_dfs[run_b],
        on=ID_COL,
        how="inner"
    )

    human_a = f"human_final__{run_a}"
    human_b = f"human_final__{run_b}"

    pred_a = f"llm_decision__{run_a}"
    pred_b = f"llm_decision__{run_b}"

    required_cols = [human_a, human_b, pred_a, pred_b]

    invalid_mask = pd.Series(False, index=df.index)

    for col in required_cols:
        invalid_mask = (
            invalid_mask
            | df[col].isna()
            | ~df[col].isin([0, 1])
        )

    df_excluded = df[invalid_mask].copy()

    df_eval = df[~invalid_mask].copy()
    df_eval[required_cols] = df_eval[required_cols].astype(int)

    inconsistent_human_mask = df_eval[human_a] != df_eval[human_b]

    if inconsistent_human_mask.any():
        inconsistent = df_eval[inconsistent_human_mask].copy()

        inconsistent.to_excel(
            OUTPUT_DIR / f"inconsistent_human_labels__{run_a}_vs_{run_b}.xlsx",
            index=False
        )

        raise ValueError(
            f"Human final labels differ between {run_a} and {run_b}. "
            f"See inconsistent_human_labels__{run_a}_vs_{run_b}.xlsx"
        )

    df_eval["human_final"] = df_eval[human_a]
    df_eval["pred_a"] = df_eval[pred_a]
    df_eval["pred_b"] = df_eval[pred_b]

    return df_eval, df_excluded


def empirical_two_sided_bootstrap_p_value(diffs):
    """
    Two-sided empirical paired-bootstrap p-value for H0: difference = 0.

    The p-value is calculated as:
        2 * min(P(bootstrap difference <= 0), P(bootstrap difference >= 0))

    A +1 finite-bootstrap correction is used so the p-value is never exactly 0.
    """
    diffs = np.asarray(diffs)

    if len(diffs) == 0:
        return np.nan

    n_valid = len(diffs)

    n_less_or_equal_zero = int((diffs <= 0).sum())
    n_greater_or_equal_zero = int((diffs >= 0).sum())

    lower_tail_probability = (n_less_or_equal_zero + 1) / (n_valid + 1)
    upper_tail_probability = (n_greater_or_equal_zero + 1) / (n_valid + 1)

    p_value = 2 * min(lower_tail_probability, upper_tail_probability)

    return min(1.0, p_value)


def bootstrap_pairwise_difference(run_a, run_b, df_pair, rng):
    y_true = df_pair["human_final"].to_numpy()
    y_pred_a = df_pair["pred_a"].to_numpy()
    y_pred_b = df_pair["pred_b"].to_numpy()

    n = len(df_pair)

    observed_a = compute_metrics(y_true, y_pred_a)
    observed_b = compute_metrics(y_true, y_pred_b)

    metric_names = ["accuracy", "recall", "precision", "cohen_kappa"]

    bootstrap_diffs = {
        metric: []
        for metric in metric_names
    }

    bootstrap_draw_rows = []

    for draw in range(1, N_BOOTSTRAP + 1):
        idx = rng.choice(n, size=n, replace=True)

        y_true_boot = y_true[idx]
        y_pred_a_boot = y_pred_a[idx]
        y_pred_b_boot = y_pred_b[idx]

        metrics_a = compute_metrics(y_true_boot, y_pred_a_boot)
        metrics_b = compute_metrics(y_true_boot, y_pred_b_boot)

        for metric in metric_names:
            value_a = metrics_a[metric]
            value_b = metrics_b[metric]

            if not np.isnan(value_a) and not np.isnan(value_b):
                diff = value_b - value_a
                bootstrap_diffs[metric].append(diff)

                if SAVE_BOOTSTRAP_DRAWS:
                    bootstrap_draw_rows.append({
                        "draw": draw,
                        "run_a": run_a,
                        "run_b": run_b,
                        "metric": metric,
                        "difference_b_minus_a": diff,
                    })

    alpha = 1 - CONFIDENCE

    rows = []

    for metric in metric_names:
        diffs = np.asarray(bootstrap_diffs[metric])

        n_valid = len(diffs)
        n_invalid = N_BOOTSTRAP - n_valid

        estimate_a = observed_a[metric]
        estimate_b = observed_b[metric]
        observed_diff = estimate_b - estimate_a

        if n_valid > 0:
            ci_lower = np.percentile(diffs, 100 * alpha / 2)
            ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))
            bootstrap_mean_diff = np.mean(diffs)
            bootstrap_sd_diff = np.std(diffs, ddof=1)

            p_value_two_sided = empirical_two_sided_bootstrap_p_value(diffs)
        else:
            ci_lower = np.nan
            ci_upper = np.nan
            bootstrap_mean_diff = np.nan
            bootstrap_sd_diff = np.nan
            p_value_two_sided = np.nan

        rows.append({
            "run_a": run_a,
            "run_b": run_b,
            "metric": metric,
            "estimate_a": estimate_a,
            "estimate_b": estimate_b,
            "observed_difference_b_minus_a": observed_diff,
            "bootstrap_mean_difference_b_minus_a": bootstrap_mean_diff,
            "bootstrap_sd_difference": bootstrap_sd_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value_two_sided": p_value_two_sided,
            "p_value_method": "two-sided empirical paired-bootstrap tail probability",
            "ci_method": "paired record-level nonparametric bootstrap percentile interval",
            "confidence": CONFIDENCE,
            "n_bootstrap": N_BOOTSTRAP,
            "n_bootstrap_valid": n_valid,
            "n_bootstrap_invalid": n_invalid,
            "n_records_compared": n,
            "n_human_include": observed_a["n_human_include"],
            "n_human_exclude": observed_a["n_human_exclude"],
            "base_rate": observed_a["base_rate"],
            "tp_a": observed_a["tp"],
            "tn_a": observed_a["tn"],
            "fp_a": observed_a["fp"],
            "fn_a": observed_a["fn"],
            "tp_b": observed_b["tp"],
            "tn_b": observed_b["tn"],
            "fp_b": observed_b["fp"],
            "fn_b": observed_b["fn"],
            "n_llm_include_a": observed_a["n_llm_include"],
            "n_llm_include_b": observed_b["n_llm_include"],
        })

    results = pd.DataFrame(rows)

    if SAVE_BOOTSTRAP_DRAWS:
        draws = pd.DataFrame(bootstrap_draw_rows)
    else:
        draws = None

    return results, draws


# --- LOAD RUNS ---
if len(MATCHED_SHEETS) < 2:
    raise ValueError("Please provide at least two runs in MATCHED_SHEETS.")

run_dfs = {
    run_name: read_run(run_name, file_path)
    for run_name, file_path in MATCHED_SHEETS.items()
}

run_names = list(MATCHED_SHEETS.keys())
pairwise_comparisons = list(combinations(run_names, 2))

print(f"Runs provided: {', '.join(run_names)}")
print(f"Number of pairwise comparisons: {len(pairwise_comparisons)}")

# --- RUN ALL PAIRWISE COMPARISONS ---
rng = np.random.default_rng(RANDOM_SEED)

all_pairwise_results = []
all_excluded_records = []
all_bootstrap_draws = []

for run_a, run_b in pairwise_comparisons:
    print(f"\nComparing {run_a} vs {run_b}...")

    df_pair, df_excluded = prepare_pair_data(
        run_a=run_a,
        run_b=run_b,
        run_dfs=run_dfs
    )

    if len(df_pair) == 0:
        raise ValueError(f"No valid paired records for {run_a} vs {run_b}.")

    df_pair.to_excel(
        OUTPUT_DIR / f"aligned_records__{run_a}_vs_{run_b}.xlsx",
        index=False
    )

    if len(df_excluded) > 0:
        df_excluded = df_excluded.copy()
        df_excluded["run_a"] = run_a
        df_excluded["run_b"] = run_b
        all_excluded_records.append(df_excluded)

        df_excluded.to_excel(
            OUTPUT_DIR / f"excluded_records__{run_a}_vs_{run_b}.xlsx",
            index=False
        )

    results, draws = bootstrap_pairwise_difference(
        run_a=run_a,
        run_b=run_b,
        df_pair=df_pair,
        rng=rng
    )

    all_pairwise_results.append(results)

    results.to_excel(
        OUTPUT_DIR / f"paired_bootstrap_differences__{run_a}_vs_{run_b}.xlsx",
        index=False
    )

    if draws is not None:
        all_bootstrap_draws.append(draws)

# --- SAVE COMBINED OUTPUTS ---
all_pairwise_results_df = pd.concat(
    all_pairwise_results,
    ignore_index=True
)

all_pairwise_results_df.to_excel(
    OUTPUT_DIR / "ALL_pairwise_paired_bootstrap_metric_differences.xlsx",
    index=False
)

if len(all_excluded_records) > 0:
    pd.concat(all_excluded_records, ignore_index=True).to_excel(
        OUTPUT_DIR / "ALL_excluded_records_missing_or_invalid_labels.xlsx",
        index=False
    )

if SAVE_BOOTSTRAP_DRAWS and len(all_bootstrap_draws) > 0:
    pd.concat(all_bootstrap_draws, ignore_index=True).to_excel(
        OUTPUT_DIR / "ALL_bootstrap_difference_draws.xlsx",
        index=False
    )

print("\nDone.")
print(f"Outputs saved in: {OUTPUT_DIR}")