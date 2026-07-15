import pandas as pd
import numpy as np
from pathlib import Path
from statistics import NormalDist

# --- CONFIG ---
ID_COL = "MesH_ID"

HUMAN_DECISIONS_FILE = "human_inclusion_decisions_only.xlsx"

HUMAN_COL_1 = "inclusion_1"
HUMAN_COL_2 = "inclusion_2"

FINAL_HUMAN_COL = "final-decision_include"
LLM_DECISION_COL = "decision_LLM_2"

OUTPUT_DIR = Path(r"human-disagreement_LLM-errors\multiple_runs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Start with one matched sheet.
# Later, you can add more runs here.
MATCHED_SHEETS = {
    "run_5_med": r"matched_sheets\matched_master_sheet_2_gpt-5-mini_bs-1.xlsx",
    "run_5_min": r"matched_sheets\matched_master_sheet_2-minimal-reasoning_gpt-5-mini_bs-1.xlsx",
    "run_5_low": r"matched_sheets\matched_master_sheet_2-low-reasoning_gpt-5-mini_bs-1.xlsx",
    "run_5_high": r"matched_sheets\matched_master_sheet_2-high-reasoning_gpt-5-mini_bs-1.xlsx",
    "run_4.1_temp1": r"matched_sheets\matched_master_sheet_2-temp1_gpt-4.1-mini_bs-1.xlsx",
    "run_4.1_temp0": r"matched_sheets\matched_master_sheet_2_gpt-4.1-mini_bs-1.xlsx",
    "run_qw3_temp1": r"matched_sheets\matched_master_sheet_2-temp1_qwen3_8b_bs-1.xlsx",
    "run_qw3_temp0": r"matched_sheets\matched_master_sheet_2_qwen3_8b_bs-1.xlsx",
}


# --- HELPERS ---
def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def risk_ratio_ci_and_p_value(a, b, c, d, confidence=0.95):
    """
    Risk ratio, CI, and p-value for a 2x2 table.

    Table layout:
        a = human disagreement + LLM error
        b = human disagreement + no LLM error
        c = no human disagreement + LLM error
        d = no human disagreement + no LLM error

    Risk ratio:
        [a / (a + b)] / [c / (c + d)]

    CI:
        Wald interval on log risk ratio scale.

    A 0.5 continuity correction is applied only if a or c is zero.
    If one exposure group has zero total records, RR is undefined.
    """
    a_raw, b_raw, c_raw, d_raw = a, b, c, d

    exposed_total = a_raw + b_raw
    unexposed_total = c_raw + d_raw

    if exposed_total == 0 or unexposed_total == 0:
        return {
            "risk_ratio_ci_lower": np.nan,
            "risk_ratio_ci_upper": np.nan,
            "risk_ratio_p_value": np.nan,
            "risk_ratio_ci_method": "log risk ratio Wald CI",
            "risk_ratio_0_5_correction_applied": False,
        }

    correction_applied = (a_raw == 0) or (c_raw == 0)

    if correction_applied:
        a = a_raw + 0.5
        b = b_raw + 0.5
        c = c_raw + 0.5
        d = d_raw + 0.5
    else:
        a = a_raw
        b = b_raw
        c = c_raw
        d = d_raw

    risk_exposed = a / (a + b)
    risk_unexposed = c / (c + d)

    if risk_unexposed == 0:
        return {
            "risk_ratio_ci_lower": np.nan,
            "risk_ratio_ci_upper": np.nan,
            "risk_ratio_p_value": np.nan,
            "risk_ratio_ci_method": "log risk ratio Wald CI",
            "risk_ratio_0_5_correction_applied": correction_applied,
        }

    rr = risk_exposed / risk_unexposed

    if rr <= 0:
        return {
            "risk_ratio_ci_lower": np.nan,
            "risk_ratio_ci_upper": np.nan,
            "risk_ratio_p_value": np.nan,
            "risk_ratio_ci_method": "log risk ratio Wald CI",
            "risk_ratio_0_5_correction_applied": correction_applied,
        }

    se_log_rr = np.sqrt(
        (1 / a)
        - (1 / (a + b))
        + (1 / c)
        - (1 / (c + d))
    )

    alpha = 1 - confidence
    z_crit = NormalDist().inv_cdf(1 - alpha / 2)

    log_rr = np.log(rr)

    ci_lower = np.exp(log_rr - z_crit * se_log_rr)
    ci_upper = np.exp(log_rr + z_crit * se_log_rr)

    z_stat = log_rr / se_log_rr
    p_value = 2 * (1 - NormalDist().cdf(abs(z_stat)))
    p_value = min(1, p_value)

    return {
        "risk_ratio_ci_lower": ci_lower,
        "risk_ratio_ci_upper": ci_upper,
        "risk_ratio_p_value": p_value,
        "risk_ratio_ci_method": "log risk ratio Wald CI",
        "risk_ratio_0_5_correction_applied": correction_applied,
    }


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

    return df


def analyze_matched_sheet(run_name, matched_sheet_path, human_df):
    matched = pd.read_excel(
        matched_sheet_path,
        usecols=[ID_COL, FINAL_HUMAN_COL, LLM_DECISION_COL]
    )

    if matched[ID_COL].duplicated().any():
        duplicated_ids = matched.loc[matched[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(
            f"Duplicate {ID_COL}s found in {run_name}: {duplicated_ids[:10]}"
        )

    matched[FINAL_HUMAN_COL] = pd.to_numeric(
        matched[FINAL_HUMAN_COL],
        errors="coerce"
    )
    matched[LLM_DECISION_COL] = pd.to_numeric(
        matched[LLM_DECISION_COL],
        errors="coerce"
    )

    # Merge original human decisions onto matched sheet
    df = matched.merge(human_df, on=ID_COL, how="inner")

    # Only keep rows with:
    # - numeric original human inclusion decisions
    # - numeric final agreed human decision
    # - numeric LLM decision
    df = df.dropna(
        subset=[
            HUMAN_COL_1,
            HUMAN_COL_2,
            FINAL_HUMAN_COL,
            LLM_DECISION_COL,
        ]
    ).copy()

    df[[FINAL_HUMAN_COL, LLM_DECISION_COL]] = df[
        [FINAL_HUMAN_COL, LLM_DECISION_COL]
    ].astype(int)

    df["run"] = run_name
    df["llm_error"] = df[LLM_DECISION_COL] != df[FINAL_HUMAN_COL]

    # Direction of LLM error
    df["error_type"] = np.select(
        [
            (df[FINAL_HUMAN_COL] == 1) & (df[LLM_DECISION_COL] == 0),
            (df[FINAL_HUMAN_COL] == 0) & (df[LLM_DECISION_COL] == 1),
            df["llm_error"] == False,
        ],
        [
            "false_exclude",
            "false_include",
            "correct",
        ],
        default="other"
    )

    # Readable group labels
    df["human_disagreement_label"] = df["human_disagreement"].map({
        False: "no_human_disagreement",
        True: "human_disagreement"
    })

    # --- SUMMARY BY HUMAN DISAGREEMENT STATUS ---
    summary = (
        df
        .groupby("human_disagreement_label")
        .agg(
            n_records=(ID_COL, "count"),
            n_llm_errors=("llm_error", "sum"),
            llm_error_rate=("llm_error", "mean"),
            n_false_exclude=("error_type", lambda x: (x == "false_exclude").sum()),
            n_false_include=("error_type", lambda x: (x == "false_include").sum()),
        )
        .reset_index()
        .rename(columns={"human_disagreement_label": "human_disagreement"})
    )

    summary["run"] = run_name

    # --- CLEAR CONTINGENCY TABLE ---
    contingency = pd.crosstab(
        df["human_disagreement"],
        df["llm_error"],
        rownames=["human_disagreement"],
        colnames=["llm_outcome"]
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
            False: "llm_correct",
            True: "llm_error",
        }
    )

    contingency_export = contingency.reset_index()
    contingency_export["run"] = run_name

    # --- ERROR TYPE BY HUMAN DISAGREEMENT STATUS ---
    error_type_summary = (
        df
        .groupby(["human_disagreement_label", "error_type"])
        .size()
        .reset_index(name="n_records")
        .rename(columns={"human_disagreement_label": "human_disagreement"})
    )

    # Denominator: all records in each human-disagreement group
    group_totals = (
        df
        .groupby("human_disagreement_label")
        .size()
        .reset_index(name="n_total_records")
        .rename(columns={"human_disagreement_label": "human_disagreement"})
    )

    error_type_summary = error_type_summary.merge(
        group_totals,
        on="human_disagreement",
        how="left"
    )

    error_type_summary["proportion_of_all_records"] = (
        error_type_summary["n_records"] / error_type_summary["n_total_records"]
    )

    # Denominator: only LLM errors in each human-disagreement group
    error_totals = (
        error_type_summary[
            error_type_summary["error_type"].isin(["false_include", "false_exclude"])
        ]
        .groupby("human_disagreement")["n_records"]
        .sum()
        .reset_index(name="n_total_llm_errors")
    )

    error_type_summary = error_type_summary.merge(
        error_totals,
        on="human_disagreement",
        how="left"
    )

    error_type_summary["proportion_of_llm_errors"] = np.where(
        error_type_summary["error_type"].isin(["false_include", "false_exclude"]),
        error_type_summary["n_records"] / error_type_summary["n_total_llm_errors"],
        np.nan
    )

    error_type_summary["run"] = run_name

    # --- ERROR TYPE AMONG ERRORS ONLY ---
    errors_only = df[df["llm_error"]].copy()

    error_type_among_errors = (
        errors_only
        .groupby(["human_disagreement_label", "error_type"])
        .size()
        .reset_index(name="n_errors")
        .rename(columns={"human_disagreement_label": "human_disagreement"})
    )

    error_type_totals = (
        error_type_among_errors
        .groupby("human_disagreement")["n_errors"]
        .sum()
        .reset_index(name="n_total_errors")
    )

    error_type_among_errors = error_type_among_errors.merge(
        error_type_totals,
        on="human_disagreement",
        how="left"
    )

    error_type_among_errors["proportion_among_errors"] = (
        error_type_among_errors["n_errors"]
        / error_type_among_errors["n_total_errors"]
    )

    error_type_among_errors["run"] = run_name

    # --- SIMPLE EFFECT MEASURES ---
    no_disagreement = df[df["human_disagreement"] == False]
    disagreement = df[df["human_disagreement"] == True]

    error_rate_no_disagreement = no_disagreement["llm_error"].mean()
    error_rate_disagreement = disagreement["llm_error"].mean()

    risk_difference = error_rate_disagreement - error_rate_no_disagreement
    risk_ratio = safe_divide(error_rate_disagreement, error_rate_no_disagreement)

    # Counts for risk ratio / odds ratio:
    # a = human disagreement + LLM error
    # b = human disagreement + no LLM error
    # c = no human disagreement + LLM error
    # d = no human disagreement + no LLM error
    a = contingency.loc["human_disagreement", "llm_error"]
    b = contingency.loc["human_disagreement", "llm_correct"]
    c = contingency.loc["no_human_disagreement", "llm_error"]
    d = contingency.loc["no_human_disagreement", "llm_correct"]

    rr_ci_p = risk_ratio_ci_and_p_value(
        a=a,
        b=b,
        c=c,
        d=d,
        confidence=0.95
    )

    # Odds ratio with small correction to avoid division-by-zero problems
    odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))

    effect_summary = pd.DataFrame([{
        "run": run_name,
        "n_records_analyzed": len(df),
        "n_human_disagreement": int(df["human_disagreement"].sum()),
        "n_no_human_disagreement": int((~df["human_disagreement"]).sum()),
        "overall_llm_error_rate": df["llm_error"].mean(),
        "llm_error_rate_human_disagreement": error_rate_disagreement,
        "llm_error_rate_no_human_disagreement": error_rate_no_disagreement,
        "risk_difference": risk_difference,
        "risk_ratio": risk_ratio,
        "risk_ratio_ci_lower": rr_ci_p["risk_ratio_ci_lower"],
        "risk_ratio_ci_upper": rr_ci_p["risk_ratio_ci_upper"],
        "risk_ratio_p_value": rr_ci_p["risk_ratio_p_value"],
        "risk_ratio_ci_method": rr_ci_p["risk_ratio_ci_method"],
        "risk_ratio_0_5_correction_applied": rr_ci_p["risk_ratio_0_5_correction_applied"],
        "odds_ratio_with_0_5_correction": odds_ratio,
    }])

    return (
        df,
        summary,
        contingency_export,
        effect_summary,
        error_type_summary,
        error_type_among_errors,
    )


# --- RUN ANALYSIS ---
human_df = read_human_decisions()

all_record_level = []
all_summaries = []
all_contingencies = []
all_effect_summaries = []
all_error_type_summaries = []
all_error_type_among_errors = []

for run_name, matched_sheet_path in MATCHED_SHEETS.items():
    print(f"\nAnalyzing {run_name}...")

    (
        record_level,
        summary,
        contingency,
        effect_summary,
        error_type_summary,
        error_type_among_errors,
    ) = analyze_matched_sheet(
        run_name=run_name,
        matched_sheet_path=matched_sheet_path,
        human_df=human_df
    )

    all_record_level.append(record_level)
    all_summaries.append(summary)
    all_contingencies.append(contingency)
    all_effect_summaries.append(effect_summary)
    all_error_type_summaries.append(error_type_summary)
    all_error_type_among_errors.append(error_type_among_errors)

    record_level.to_excel(
        OUTPUT_DIR / f"{run_name}_record_level_human_disagreement_llm_errors.xlsx",
        index=False
    )

    summary.to_excel(
        OUTPUT_DIR / f"{run_name}_summary_by_human_disagreement.xlsx",
        index=False
    )

    contingency.to_excel(
        OUTPUT_DIR / f"{run_name}_contingency_table.xlsx",
        index=False
    )

    effect_summary.to_excel(
        OUTPUT_DIR / f"{run_name}_effect_summary.xlsx",
        index=False
    )

    error_type_summary.to_excel(
        OUTPUT_DIR / f"{run_name}_error_type_by_human_disagreement.xlsx",
        index=False
    )

    error_type_among_errors.to_excel(
        OUTPUT_DIR / f"{run_name}_error_type_among_errors_only.xlsx",
        index=False
    )

# --- COMBINED OUTPUTS, USEFUL ONCE MULTIPLE RUNS ARE ADDED ---
pd.concat(all_record_level, ignore_index=True).to_excel(
    OUTPUT_DIR / "ALL_record_level_human_disagreement_llm_errors.xlsx",
    index=False
)

pd.concat(all_summaries, ignore_index=True).to_excel(
    OUTPUT_DIR / "ALL_summary_by_human_disagreement.xlsx",
    index=False
)

pd.concat(all_contingencies, ignore_index=True).to_excel(
    OUTPUT_DIR / "ALL_contingency_tables.xlsx",
    index=False
)

pd.concat(all_effect_summaries, ignore_index=True).to_excel(
    OUTPUT_DIR / "ALL_effect_summaries.xlsx",
    index=False
)

pd.concat(all_error_type_summaries, ignore_index=True).to_excel(
    OUTPUT_DIR / "ALL_error_type_by_human_disagreement.xlsx",
    index=False
)

pd.concat(all_error_type_among_errors, ignore_index=True).to_excel(
    OUTPUT_DIR / "ALL_error_type_among_errors_only.xlsx",
    index=False
)

print("\nDone.")
print(f"Outputs saved in: {OUTPUT_DIR}")