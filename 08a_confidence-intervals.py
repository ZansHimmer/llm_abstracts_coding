import pandas as pd
import numpy as np
from pathlib import Path
from statistics import NormalDist

# --- CONFIG ---
ID_COL = "MesH_ID"
FINAL_HUMAN_COL = "final-decision_include"
LLM_DECISION_COL = "decision_LLM_2"

OUTPUT_DIR = Path(r"screening_metrics_with_CI\gpt-4.1-mini_temp_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHED_SHEETS = {
    "run_1_temp0": r"matched_sheets\matched_master_sheet_2_gpt-4.1-mini_bs-1.xlsx",
    "run_2_temp1": r"matched_sheets\matched_master_sheet_2-temp1_gpt-4.1-mini_bs-1.xlsx",
    "run_2b_temp1": r"matched_sheets\matched_master_sheet_2b-temp1_gpt-4.1-mini_bs-1.xlsx",
    "run_2c_temp1": r"matched_sheets\matched_master_sheet_2c-temp1_gpt-4.1-mini_bs-1.xlsx",
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


def wilson_ci(successes, n, confidence=0.95):
    """
    Wilson score interval for a binomial proportion.
    """
    if n == 0:
        return np.nan, np.nan

    alpha = 1 - confidence
    z = NormalDist().inv_cdf(1 - alpha / 2)

    p_hat = successes / n

    denominator = 1 + z**2 / n

    center = (
        p_hat + z**2 / (2 * n)
    ) / denominator

    margin = (
        z
        * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))
        / denominator
    )

    lower = center - margin
    upper = center + margin

    return lower, upper


def cohen_kappa_binary(y_true, y_pred):
    """
    Cohen's kappa for binary 0/1 labels.
    Returns np.nan if kappa is undefined.
    """
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


def bootstrap_kappa_ci(
    y_true,
    y_pred,
    n_bootstrap=5000,
    confidence=0.95,
    seed=123
):
    """
    Record-level nonparametric bootstrap CI for Cohen's kappa.
    Resamples records with replacement and recomputes kappa.
    """
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)

    observed_kappa = cohen_kappa_binary(y_true, y_pred)

    bootstrap_kappas = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)

        kappa = cohen_kappa_binary(
            y_true[idx],
            y_pred[idx]
        )

        if not np.isnan(kappa):
            bootstrap_kappas.append(kappa)

    bootstrap_kappas = np.asarray(bootstrap_kappas)

    n_valid = len(bootstrap_kappas)
    n_invalid = n_bootstrap - n_valid

    if n_valid == 0:
        return observed_kappa, np.nan, np.nan, n_valid, n_invalid, bootstrap_kappas

    alpha = 1 - confidence

    ci_lower = np.percentile(bootstrap_kappas, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_kappas, 100 * (1 - alpha / 2))

    return observed_kappa, ci_lower, ci_upper, n_valid, n_invalid, bootstrap_kappas


def read_matched_sheet(file_path):
    df = pd.read_excel(
        file_path,
        usecols=[ID_COL, FINAL_HUMAN_COL, LLM_DECISION_COL]
    )

    if df[ID_COL].duplicated().any():
        df = df.drop_duplicates(subset=[ID_COL], keep="last").copy()

    df[FINAL_HUMAN_COL] = pd.to_numeric(df[FINAL_HUMAN_COL], errors="coerce")
    df[LLM_DECISION_COL] = pd.to_numeric(df[LLM_DECISION_COL], errors="coerce")

    # Only keep rows with numeric human and LLM decisions
    df = df.dropna(
        subset=[FINAL_HUMAN_COL, LLM_DECISION_COL]
    ).copy()

    df[[FINAL_HUMAN_COL, LLM_DECISION_COL]] = df[
        [FINAL_HUMAN_COL, LLM_DECISION_COL]
    ].astype(int)

    # Only keep binary labels
    df = df[
        df[FINAL_HUMAN_COL].isin([0, 1])
        & df[LLM_DECISION_COL].isin([0, 1])
    ].copy()

    return df


def compute_confusion_counts(df):
    y_true = df[FINAL_HUMAN_COL].astype(int)
    y_pred = df[LLM_DECISION_COL].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return tp, tn, fp, fn


def analyze_run(run_name, file_path):
    df = read_matched_sheet(file_path)

    tp, tn, fp, fn = compute_confusion_counts(df)

    n_records = tp + tn + fp + fn
    n_human_include = tp + fn
    n_human_exclude = tn + fp
    n_llm_include = tp + fp
    n_llm_exclude = tn + fn

    base_rate = safe_divide(n_human_include, n_records)

    # --- Point estimates ---
    accuracy = safe_divide(tp + tn, n_records)
    recall = safe_divide(tp, tp + fn)
    precision = safe_divide(tp, tp + fp)

    # --- Wilson CIs ---
    accuracy_ci_lower, accuracy_ci_upper = wilson_ci(
        successes=tp + tn,
        n=n_records,
        confidence=CONFIDENCE
    )

    recall_ci_lower, recall_ci_upper = wilson_ci(
        successes=tp,
        n=tp + fn,
        confidence=CONFIDENCE
    )

    precision_ci_lower, precision_ci_upper = wilson_ci(
        successes=tp,
        n=tp + fp,
        confidence=CONFIDENCE
    )

    # --- Bootstrap CI for Cohen's kappa ---
    (
        kappa,
        kappa_ci_lower,
        kappa_ci_upper,
        kappa_bootstrap_valid,
        kappa_bootstrap_invalid,
        bootstrap_kappas,
    ) = bootstrap_kappa_ci(
        y_true=df[FINAL_HUMAN_COL],
        y_pred=df[LLM_DECISION_COL],
        n_bootstrap=N_BOOTSTRAP,
        confidence=CONFIDENCE,
        seed=RANDOM_SEED
    )

    # --- Clean metric table ---
    metrics = pd.DataFrame([
        {
            "run": run_name,
            "metric": "accuracy",
            "estimate": accuracy,
            "ci_lower": accuracy_ci_lower,
            "ci_upper": accuracy_ci_upper,
            "ci_method": "Wilson score interval",
            "successes": tp + tn,
            "denominator": n_records,
            "n_records": n_records,
            "n_human_include": n_human_include,
            "n_human_exclude": n_human_exclude,
            "base_rate": base_rate,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n_bootstrap": np.nan,
            "n_bootstrap_valid": np.nan,
            "n_bootstrap_invalid": np.nan,
        },
        {
            "run": run_name,
            "metric": "recall",
            "estimate": recall,
            "ci_lower": recall_ci_lower,
            "ci_upper": recall_ci_upper,
            "ci_method": "Wilson score interval",
            "successes": tp,
            "denominator": tp + fn,
            "n_records": n_records,
            "n_human_include": n_human_include,
            "n_human_exclude": n_human_exclude,
            "base_rate": base_rate,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n_bootstrap": np.nan,
            "n_bootstrap_valid": np.nan,
            "n_bootstrap_invalid": np.nan,
        },
        {
            "run": run_name,
            "metric": "precision",
            "estimate": precision,
            "ci_lower": precision_ci_lower,
            "ci_upper": precision_ci_upper,
            "ci_method": "Wilson score interval",
            "successes": tp,
            "denominator": tp + fp,
            "n_records": n_records,
            "n_human_include": n_human_include,
            "n_human_exclude": n_human_exclude,
            "base_rate": base_rate,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n_bootstrap": np.nan,
            "n_bootstrap_valid": np.nan,
            "n_bootstrap_invalid": np.nan,
        },
        {
            "run": run_name,
            "metric": "cohen_kappa",
            "estimate": kappa,
            "ci_lower": kappa_ci_lower,
            "ci_upper": kappa_ci_upper,
            "ci_method": "record-level nonparametric bootstrap percentile interval",
            "successes": np.nan,
            "denominator": n_records,
            "n_records": n_records,
            "n_human_include": n_human_include,
            "n_human_exclude": n_human_exclude,
            "base_rate": base_rate,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n_bootstrap": N_BOOTSTRAP,
            "n_bootstrap_valid": kappa_bootstrap_valid,
            "n_bootstrap_invalid": kappa_bootstrap_invalid,
        },
    ])

    # Optional: compact counts table
    counts = pd.DataFrame([{
        "run": run_name,
        "n_records": n_records,
        "n_human_include": n_human_include,
        "n_human_exclude": n_human_exclude,
        "base_rate": base_rate,
        "n_llm_include": n_llm_include,
        "n_llm_exclude": n_llm_exclude,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }])

    if SAVE_BOOTSTRAP_DRAWS:
        bootstrap_df = pd.DataFrame({
            "run": run_name,
            "bootstrap_draw": np.arange(1, len(bootstrap_kappas) + 1),
            "kappa": bootstrap_kappas,
        })
    else:
        bootstrap_df = None

    return metrics, counts, bootstrap_df


# --- RUN ANALYSIS ---
all_metrics = []
all_counts = []
all_bootstrap_draws = []

for run_name, file_path in MATCHED_SHEETS.items():
    print(f"\nAnalyzing {run_name}...")

    metrics, counts, bootstrap_df = analyze_run(run_name, file_path)

    all_metrics.append(metrics)
    all_counts.append(counts)

    metrics.to_excel(
        OUTPUT_DIR / f"{run_name}_screening_metrics_with_CI.xlsx",
        index=False
    )

    counts.to_excel(
        OUTPUT_DIR / f"{run_name}_confusion_counts.xlsx",
        index=False
    )

    if bootstrap_df is not None:
        all_bootstrap_draws.append(bootstrap_df)

        bootstrap_df.to_excel(
            OUTPUT_DIR / f"{run_name}_kappa_bootstrap_draws.xlsx",
            index=False
        )

# --- COMBINED OUTPUTS ---
all_metrics_df = pd.concat(all_metrics, ignore_index=True)
all_counts_df = pd.concat(all_counts, ignore_index=True)

all_metrics_df.to_excel(
    OUTPUT_DIR / "ALL_screening_metrics_with_CI.xlsx",
    index=False
)

all_counts_df.to_excel(
    OUTPUT_DIR / "ALL_confusion_counts.xlsx",
    index=False
)

if SAVE_BOOTSTRAP_DRAWS and len(all_bootstrap_draws) > 0:
    pd.concat(all_bootstrap_draws, ignore_index=True).to_excel(
        OUTPUT_DIR / "ALL_kappa_bootstrap_draws.xlsx",
        index=False
    )

print("\nDone.")
print(f"Outputs saved in: {OUTPUT_DIR}")