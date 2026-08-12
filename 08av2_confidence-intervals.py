import pandas as pd
import numpy as np
from pathlib import Path


# --- CONFIG ---

ID_COL = "MesH_ID"
FINAL_HUMAN_COL = "final-decision_include"
LLM_DECISION_COL = "decision_LLM_2"

OUTPUT_DIR = Path(r"screening_metrics_with_CI_v2\gpt-4.1-mini_bs-5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHED_SHEETS = {
    "run_1_temp0": {
        "path": r"matched_sheets\matched_master_sheet_2_gpt-4.1-mini_bs-5.xlsx",
        "batch_size": 5,
    },
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


def compute_metrics_from_counts(tp, tn, fp, fn):

    n = tp + tn + fp + fn

    accuracy = safe_divide(tp + tn, n)
    recall = safe_divide(tp, tp + fn)
    precision = safe_divide(tp, tp + fp)

    observed_agreement = accuracy

    p_true_1 = safe_divide(tp + fn, n)
    p_true_0 = safe_divide(tn + fp, n)
    p_pred_1 = safe_divide(tp + fp, n)
    p_pred_0 = safe_divide(tn + fn, n)

    expected_agreement = (
        p_true_1 * p_pred_1
        + p_true_0 * p_pred_0
    )

    if (
        pd.isna(expected_agreement)
        or expected_agreement == 1
    ):
        kappa = np.nan
    else:
        kappa = (
            observed_agreement - expected_agreement
        ) / (
            1 - expected_agreement
        )

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "cohen_kappa": kappa,
    }


def read_matched_sheet(file_path, batch_size):

    df = pd.read_excel(
        file_path,
        usecols=[
            ID_COL,
            FINAL_HUMAN_COL,
            LLM_DECISION_COL,
        ]
    )

    df = df.iloc[:5000].copy()
    df = df.reset_index(drop=True)

    if df[ID_COL].duplicated().any():
        df = df.drop_duplicates(
            subset=[ID_COL],
            keep="last"
        ).copy()

    df = df.reset_index(drop=True)

    if batch_size > 1:
        df["prompt_id"] = (
            np.arange(len(df)) // batch_size
        )

    return df


def clean_valid_records(df):

    df = df.copy()

    df[FINAL_HUMAN_COL] = pd.to_numeric(
        df[FINAL_HUMAN_COL],
        errors="coerce"
    )

    df[LLM_DECISION_COL] = pd.to_numeric(
        df[LLM_DECISION_COL],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            FINAL_HUMAN_COL,
            LLM_DECISION_COL,
        ]
    ).copy()

    df[[FINAL_HUMAN_COL, LLM_DECISION_COL]] = df[
        [FINAL_HUMAN_COL, LLM_DECISION_COL]
    ].astype(int)

    df = df[
        df[FINAL_HUMAN_COL].isin([0, 1])
        & df[LLM_DECISION_COL].isin([0, 1])
    ].copy()

    return df


def add_confusion_columns(df):

    df = df.copy()

    y_true = df[FINAL_HUMAN_COL]
    y_pred = df[LLM_DECISION_COL]

    df["tp"] = (
        (y_true == 1) & (y_pred == 1)
    ).astype(int)

    df["tn"] = (
        (y_true == 0) & (y_pred == 0)
    ).astype(int)

    df["fp"] = (
        (y_true == 0) & (y_pred == 1)
    ).astype(int)

    df["fn"] = (
        (y_true == 1) & (y_pred == 0)
    ).astype(int)

    return df


def prepare_resampling_units(raw_df, valid_df, batch_size):

    valid_df = add_confusion_columns(valid_df)

    if batch_size == 1:

        unit_counts = valid_df[
            ["tp", "tn", "fp", "fn"]
        ].to_numpy()

    else:

        all_prompt_ids = raw_df[
            "prompt_id"
        ].drop_duplicates()

        prompt_counts = (
            valid_df
            .groupby("prompt_id")[
                ["tp", "tn", "fp", "fn"]
            ]
            .sum()
            .reindex(
                all_prompt_ids,
                fill_value=0
            )
        )

        unit_counts = prompt_counts.to_numpy()

    return unit_counts


def bootstrap_metric_cis(
    unit_counts,
    n_bootstrap=5000,
    confidence=0.95,
    seed=123
):

    rng = np.random.default_rng(seed)

    n_units = len(unit_counts)

    bootstrap_values = {
        "accuracy": [],
        "recall": [],
        "precision": [],
        "cohen_kappa": [],
    }

    for _ in range(n_bootstrap):

        idx = rng.choice(
            n_units,
            size=n_units,
            replace=True
        )

        counts = unit_counts[idx].sum(axis=0)

        tp, tn, fp, fn = counts

        metrics = compute_metrics_from_counts(
            tp, tn, fp, fn
        )

        for metric, value in metrics.items():
            if not np.isnan(value):
                bootstrap_values[metric].append(value)

    alpha = 1 - confidence

    results = {}

    for metric, values in bootstrap_values.items():

        values = np.asarray(values)

        if len(values) == 0:
            lower = np.nan
            upper = np.nan
        else:
            lower = np.percentile(
                values,
                100 * alpha / 2
            )
            upper = np.percentile(
                values,
                100 * (1 - alpha / 2)
            )

        results[metric] = {
            "ci_lower": lower,
            "ci_upper": upper,
            "n_valid": len(values),
            "n_invalid": n_bootstrap - len(values),
            "draws": values,
        }

    return results


def analyze_run(
    run_name,
    file_path,
    batch_size
):

    raw_df = read_matched_sheet(
        file_path,
        batch_size
    )

    df = clean_valid_records(raw_df)
    df = add_confusion_columns(df)

    tp = int(df["tp"].sum())
    tn = int(df["tn"].sum())
    fp = int(df["fp"].sum())
    fn = int(df["fn"].sum())

    n_records = tp + tn + fp + fn

    n_human_include = tp + fn
    n_human_exclude = tn + fp

    n_llm_include = tp + fp
    n_llm_exclude = tn + fn

    base_rate = safe_divide(
        n_human_include,
        n_records
    )

    point_estimates = compute_metrics_from_counts(
        tp, tn, fp, fn
    )

    unit_counts = prepare_resampling_units(
        raw_df,
        df,
        batch_size
    )

    bootstrap_results = bootstrap_metric_cis(
        unit_counts=unit_counts,
        n_bootstrap=N_BOOTSTRAP,
        confidence=CONFIDENCE,
        seed=RANDOM_SEED
    )

    if batch_size == 1:
        ci_method = (
            "record-level nonparametric "
            "bootstrap percentile interval"
        )
    else:
        ci_method = (
            "prompt-level cluster bootstrap "
            "percentile interval"
        )

    rows = []

    for metric in [
        "accuracy",
        "recall",
        "precision",
        "cohen_kappa",
    ]:

        rows.append({
            "run": run_name,
            "batch_size": batch_size,
            "metric": metric,

            "estimate":
                point_estimates[metric],

            "ci_lower":
                bootstrap_results[metric]["ci_lower"],

            "ci_upper":
                bootstrap_results[metric]["ci_upper"],

            "ci_method":
                ci_method,

            "n_records":
                n_records,

            "n_resampling_units":
                len(unit_counts),

            "n_human_include":
                n_human_include,

            "n_human_exclude":
                n_human_exclude,

            "base_rate":
                base_rate,

            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,

            "n_bootstrap":
                N_BOOTSTRAP,

            "n_bootstrap_valid":
                bootstrap_results[metric]["n_valid"],

            "n_bootstrap_invalid":
                bootstrap_results[metric]["n_invalid"],
        })

    metrics = pd.DataFrame(rows)

    counts = pd.DataFrame([{
        "run": run_name,
        "batch_size": batch_size,
        "n_records": n_records,
        "n_resampling_units": len(unit_counts),
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
            "batch_size": batch_size,
            "bootstrap_draw":
                np.arange(1, N_BOOTSTRAP + 1),
        })

        for metric in [
            "accuracy",
            "recall",
            "precision",
            "cohen_kappa",
        ]:
            values = bootstrap_results[
                metric
            ]["draws"]

            if len(values) == N_BOOTSTRAP:
                bootstrap_df[metric] = values

    else:
        bootstrap_df = None

    return metrics, counts, bootstrap_df


# --- RUN ANALYSIS ---

all_metrics = []
all_counts = []
all_bootstrap_draws = []

for run_name, config in MATCHED_SHEETS.items():

    print(f"\nAnalyzing {run_name}...")

    metrics, counts, bootstrap_df = analyze_run(
        run_name=run_name,
        file_path=config["path"],
        batch_size=config["batch_size"]
    )

    all_metrics.append(metrics)
    all_counts.append(counts)

    metrics.to_excel(
        OUTPUT_DIR
        / f"{run_name}_screening_metrics_with_CI.xlsx",
        index=False
    )

    counts.to_excel(
        OUTPUT_DIR
        / f"{run_name}_confusion_counts.xlsx",
        index=False
    )

    if bootstrap_df is not None:

        all_bootstrap_draws.append(
            bootstrap_df
        )

        bootstrap_df.to_excel(
            OUTPUT_DIR
            / f"{run_name}_bootstrap_draws.xlsx",
            index=False
        )


# --- COMBINED OUTPUTS ---

all_metrics_df = pd.concat(
    all_metrics,
    ignore_index=True
)

all_counts_df = pd.concat(
    all_counts,
    ignore_index=True
)

all_metrics_df.to_excel(
    OUTPUT_DIR
    / "ALL_screening_metrics_with_CI.xlsx",
    index=False
)

all_counts_df.to_excel(
    OUTPUT_DIR
    / "ALL_confusion_counts.xlsx",
    index=False
)

if (
    SAVE_BOOTSTRAP_DRAWS
    and len(all_bootstrap_draws) > 0
):
    pd.concat(
        all_bootstrap_draws,
        ignore_index=True
    ).to_excel(
        OUTPUT_DIR
        / "ALL_bootstrap_draws.xlsx",
        index=False
    )


print("\nDone.")
print(f"Outputs saved in: {OUTPUT_DIR}")