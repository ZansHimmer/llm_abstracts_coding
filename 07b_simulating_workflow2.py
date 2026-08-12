import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

ID_COL = "MesH_ID"
FINAL_HUMAN_COL = "final-decision_include"
LLM_DECISION_COL = "decision_LLM_2"

BATCH_SIZE = 100

OUTPUT_DIR = Path(
    rf"subsample_screening_performance_v2\gpt-5-mini_minimal_bs-{BATCH_SIZE}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MATCHED_SHEETS = {
    "run_1": rf"matched_sheets\matched_master_sheet_2-minimal-reasoning_gpt-5-mini_bs-{BATCH_SIZE}.xlsx",
}


SAMPLE_PROPORTIONS = [0.10, 0.15, 0.20]

N_DRAWS = 1000
RANDOM_SEED = 123

STRATIFY_BY_HUMAN_LABEL = False

KAPPA_THRESHOLD = 0.80
RECALL_THRESHOLD = 0.90


# ============================================================
# HELPERS
# ============================================================

def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def compute_metrics(df):
    """
    Computes screening performance for binary labels.
    Final human decision = ground truth
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

    if (
        pd.isna(precision)
        or pd.isna(recall)
        or precision + recall == 0
    ):
        f1 = np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)

    observed_agreement = accuracy

    p_true_include = safe_divide(tp + fn, n)
    p_true_exclude = safe_divide(tn + fp, n)
    p_pred_include = safe_divide(tp + fp, n)
    p_pred_exclude = safe_divide(tn + fn, n)

    expected_agreement = (
        p_true_include * p_pred_include
        + p_true_exclude * p_pred_exclude
    )

    if pd.isna(expected_agreement) or expected_agreement == 1:
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


def clean_valid_records(df):
    """
    Removes records without valid binary human/LLM decisions.

    For batch sizes > 1, this is called only after prompt IDs
    have already been assigned.
    """

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
        subset=[FINAL_HUMAN_COL, LLM_DECISION_COL]
    ).copy()

    df[[FINAL_HUMAN_COL, LLM_DECISION_COL]] = df[
        [FINAL_HUMAN_COL, LLM_DECISION_COL]
    ].astype(int)

    df = df[
        df[FINAL_HUMAN_COL].isin([0, 1])
        & df[LLM_DECISION_COL].isin([0, 1])
    ].copy()

    return df


def read_matched_sheet(file_path):
    """
    Reads the matched sheet.

    For batch sizes > 1, prompt IDs are assigned before
    invalid records are removed. Thus, records with missing
    human decisions still retain their original place in the
    prompt structure.
    """

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

    duplicate_mask = df[ID_COL].duplicated(
        keep="last"
    )

    if duplicate_mask.any():
        removed_duplicate_ids = df.loc[
            duplicate_mask,
            ID_COL
        ].tolist()

        print(
            f"Removing {duplicate_mask.sum()} earlier rows "
            f"for repeated {ID_COL} values. "
            f"Examples: {removed_duplicate_ids[:10]}"
        )

        df = df.loc[
            ~duplicate_mask
        ].copy()

        df = df.reset_index(drop=True)

    if BATCH_SIZE > 1:
        df["prompt_id"] = (
            np.arange(len(df)) // BATCH_SIZE
        )

    return df


def draw_sample(df, sample_proportion, random_state):
    """
    Draw evaluation sample and remainder.

    BATCH_SIZE = 1:
        Clean records first, then randomly sample records.

    BATCH_SIZE > 1:
        Prompt IDs already exist based on the original
        record order. Entire prompts are sampled.
        Invalid records are removed only afterwards.
    """

    if BATCH_SIZE == 1:

        valid_df = clean_valid_records(df)

        if STRATIFY_BY_HUMAN_LABEL:
            sample = (
                valid_df
                .groupby(
                    FINAL_HUMAN_COL,
                    group_keys=False
                )
                .sample(
                    frac=sample_proportion,
                    random_state=random_state
                )
            )
        else:
            sample = valid_df.sample(
                frac=sample_proportion,
                random_state=random_state
            )

        remainder = valid_df.drop(
            index=sample.index
        )

        return sample, remainder

    prompt_ids = df["prompt_id"].drop_duplicates()

    sampled_prompt_ids = prompt_ids.sample(
        frac=sample_proportion,
        random_state=random_state
    )

    sample_raw = df[
        df["prompt_id"].isin(sampled_prompt_ids)
    ].copy()

    remainder_raw = df[
        ~df["prompt_id"].isin(sampled_prompt_ids)
    ].copy()

    sample = clean_valid_records(sample_raw)
    remainder = clean_valid_records(remainder_raw)

    return sample, remainder


def add_metric_prefix(metrics, prefix):
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
    }


# ============================================================
# SUMMARY OF SAMPLE VS. REMAINDER
# ============================================================

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

            diff_vs_remainder = (
                group[sample_col]
                - group[remainder_col]
            )

            abs_diff_vs_remainder = (
                diff_vs_remainder.abs()
            )

            diff_vs_full = (
                group[sample_col]
                - group[full_col]
            )

            abs_diff_vs_full = (
                diff_vs_full.abs()
            )

            rows.append({
                "run": run,
                "sample_proportion": sample_proportion,
                "metric": metric,

                "full_metric":
                    group[full_col].iloc[0],

                "mean_sample_metric":
                    group[sample_col].mean(),

                "sd_sample_metric":
                    group[sample_col].std(),

                "mean_remainder_metric":
                    group[remainder_col].mean(),

                "sd_remainder_metric":
                    group[remainder_col].std(),

                "mean_difference_sample_minus_remainder":
                    diff_vs_remainder.mean(),

                "mean_abs_difference_sample_vs_remainder":
                    abs_diff_vs_remainder.mean(),

                "p95_abs_difference_sample_vs_remainder":
                    abs_diff_vs_remainder.quantile(0.95),

                "difference_vs_remainder_q025":
                    diff_vs_remainder.quantile(0.025),

                "difference_vs_remainder_q975":
                    diff_vs_remainder.quantile(0.975),

                "mean_abs_difference_sample_vs_full":
                    abs_diff_vs_full.mean(),

                "p95_abs_difference_sample_vs_full":
                    abs_diff_vs_full.quantile(0.95),
            })

    return pd.DataFrame(rows)


# ============================================================
# WORKFLOW DECISION RULE
# ============================================================

def add_workflow_decision_variables(sim_df):

    df = sim_df.copy()

    df["sample_recall_pass"] = (
        df["sample_recall"].notna()
        & (df["sample_recall"] >= RECALL_THRESHOLD)
    )

    df["sample_kappa_pass"] = (
        df["sample_kappa"].notna()
        & (df["sample_kappa"] >= KAPPA_THRESHOLD)
    )

    df["sample_pass"] = (
        df["sample_recall_pass"]
        & df["sample_kappa_pass"]
    )

    df["remainder_recall_pass"] = (
        df["remainder_recall"].notna()
        & (df["remainder_recall"] >= RECALL_THRESHOLD)
    )

    df["remainder_kappa_pass"] = (
        df["remainder_kappa"].notna()
        & (df["remainder_kappa"] >= KAPPA_THRESHOLD)
    )

    df["remainder_pass"] = (
        df["remainder_recall_pass"]
        & df["remainder_kappa_pass"]
    )

    return df


def summarize_decision_rule(sim_df):

    rows = []

    for (run, sample_proportion), group in sim_df.groupby(
        ["run", "sample_proportion"]
    ):

        sample_pass = group["sample_pass"]
        remainder_pass = group["remainder_pass"]

        both_pass = (
            sample_pass & remainder_pass
        ).sum()

        sample_pass_remainder_fail = (
            sample_pass & ~remainder_pass
        ).sum()

        sample_fail_remainder_pass = (
            ~sample_pass & remainder_pass
        ).sum()

        both_fail = (
            ~sample_pass & ~remainder_pass
        ).sum()

        n = len(group)
        n_sample_pass = sample_pass.sum()

        recall_fail_after_sample_pass = (
            sample_pass
            & ~group["remainder_recall_pass"]
        ).sum()

        kappa_fail_after_sample_pass = (
            sample_pass
            & ~group["remainder_kappa_pass"]
        ).sum()

        both_thresholds_fail_after_sample_pass = (
            sample_pass
            & ~group["remainder_recall_pass"]
            & ~group["remainder_kappa_pass"]
        ).sum()

        rows.append({
            "run": run,
            "sample_proportion": sample_proportion,

            "kappa_threshold":
                KAPPA_THRESHOLD,

            "recall_threshold":
                RECALL_THRESHOLD,

            "n_draws":
                n,

            "sample_pass_n":
                int(sample_pass.sum()),

            "sample_pass_rate":
                sample_pass.mean(),

            "sample_fail_n":
                int((~sample_pass).sum()),

            "remainder_pass_n":
                int(remainder_pass.sum()),

            "remainder_pass_rate":
                remainder_pass.mean(),

            "remainder_fail_n":
                int((~remainder_pass).sum()),

            "both_pass_n":
                int(both_pass),

            "both_pass_rate":
                both_pass / n,

            "sample_pass_remainder_fail_n":
                int(sample_pass_remainder_fail),

            "sample_pass_remainder_fail_rate_all_draws":
                sample_pass_remainder_fail / n,

            "sample_fail_remainder_pass_n":
                int(sample_fail_remainder_pass),

            "sample_fail_remainder_pass_rate_all_draws":
                sample_fail_remainder_pass / n,

            "both_fail_n":
                int(both_fail),

            "both_fail_rate":
                both_fail / n,

            "remainder_pass_given_sample_pass":
                safe_divide(
                    both_pass,
                    n_sample_pass
                ),

            "remainder_fail_given_sample_pass":
                safe_divide(
                    sample_pass_remainder_fail,
                    n_sample_pass
                ),

            "remainder_recall_fail_after_sample_pass_n":
                int(recall_fail_after_sample_pass),

            "remainder_recall_fail_given_sample_pass":
                safe_divide(
                    recall_fail_after_sample_pass,
                    n_sample_pass
                ),

            "remainder_kappa_fail_after_sample_pass_n":
                int(kappa_fail_after_sample_pass),

            "remainder_kappa_fail_given_sample_pass":
                safe_divide(
                    kappa_fail_after_sample_pass,
                    n_sample_pass
                ),

            "remainder_both_thresholds_fail_after_sample_pass_n":
                int(both_thresholds_fail_after_sample_pass),

            "remainder_both_thresholds_fail_given_sample_pass":
                safe_divide(
                    both_thresholds_fail_after_sample_pass,
                    n_sample_pass
                ),
        })

    return pd.DataFrame(rows)


# ============================================================
# RUN ANALYSIS
# ============================================================

all_full_metrics = []
all_simulation_draws = []

rng = np.random.default_rng(
    RANDOM_SEED
)


for run_name, matched_sheet_path in MATCHED_SHEETS.items():

    print(f"\nAnalyzing {run_name}...")

    raw_df = read_matched_sheet(
        matched_sheet_path
    )

    valid_full_df = clean_valid_records(
        raw_df
    )

    print(
        f"Original records: {len(raw_df)}"
    )

    print(
        f"Valid records available for analysis: "
        f"{len(valid_full_df)}"
    )

    if BATCH_SIZE > 1:
        print(
            f"Original prompts: "
            f"{raw_df['prompt_id'].nunique()}"
        )

    full_metrics = compute_metrics(
        valid_full_df
    )

    all_full_metrics.append({
        "run": run_name,
        **full_metrics
    })

    for sample_proportion in SAMPLE_PROPORTIONS:

        print(
            f"  Simulating sample proportion: "
            f"{sample_proportion}"
        )

        for draw in range(
            1,
            N_DRAWS + 1
        ):

            random_state = int(
                rng.integers(
                    0,
                    2**32 - 1
                )
            )

            sample, remainder = draw_sample(
                df=raw_df,
                sample_proportion=sample_proportion,
                random_state=random_state
            )

            sample_metrics = compute_metrics(
                sample
            )

            remainder_metrics = compute_metrics(
                remainder
            )

            row = {
                "run": run_name,
                "draw": draw,
                "batch_size": BATCH_SIZE,
                "sample_proportion":
                    sample_proportion,
                "n_total":
                    len(valid_full_df),
                "n_sample":
                    len(sample),
                "n_remainder":
                    len(remainder),
            }

            row.update(
                add_metric_prefix(
                    full_metrics,
                    "full"
                )
            )

            row.update(
                add_metric_prefix(
                    sample_metrics,
                    "sample"
                )
            )

            row.update(
                add_metric_prefix(
                    remainder_metrics,
                    "remainder"
                )
            )

            all_simulation_draws.append(
                row
            )


# ============================================================
# OUTPUTS
# ============================================================

full_metrics_df = pd.DataFrame(
    all_full_metrics
)

simulation_draws_df = pd.DataFrame(
    all_simulation_draws
)

simulation_draws_df = (
    add_workflow_decision_variables(
        simulation_draws_df
    )
)

simulation_summary_df = (
    summarize_simulations(
        simulation_draws_df
    )
)

decision_rule_summary_df = (
    summarize_decision_rule(
        simulation_draws_df
    )
)


full_metrics_df.to_excel(
    OUTPUT_DIR
    / "full_dataset_screening_metrics.xlsx",
    index=False
)

simulation_draws_df.to_excel(
    OUTPUT_DIR
    / "subsample_simulation_draws.xlsx",
    index=False
)

simulation_summary_df.to_excel(
    OUTPUT_DIR
    / "subsample_vs_remainder_summary.xlsx",
    index=False
)

decision_rule_summary_df.to_excel(
    OUTPUT_DIR
    / "workflow_decision_rule_summary.xlsx",
    index=False
)


print("\nDone.")
print(f"Outputs saved in: {OUTPUT_DIR}")

print("\nWorkflow decision-rule summary:")

print(
    decision_rule_summary_df[
        [
            "run",
            "sample_proportion",
            "sample_pass_rate",
            "remainder_pass_rate",
            "remainder_pass_given_sample_pass",
            "remainder_fail_given_sample_pass",
        ]
    ].to_string(index=False)
)