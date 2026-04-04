"""
master_config.py

Single source of truth for your experiment parameters.

Edit ONLY the section "EDIT THESE" for normal use.
All paths (outputs/responses/matched sheets/cost files) are derived automatically
from a few core settings: experiment id/variant, model, temperature, reasoning effort, batch size.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional


# =========================
# EDIT THESE (core settings)
# =========================

# Experiment identifier (e.g., "2", "8", "9") and optional run variant ("", "b", "c" ...)
EXPERIMENT_ID: str = "2"
RUN_VARIANT: str = ""  # e.g. "", "b", "c"

# Model & decoding
MODEL: str = "gpt-5-mini"   # for local Ollama, e.g. "qwen3:8b"
TEMPERATURE: Optional[float] = None  # None => do not send temperature (keeps current API behavior)
REASONING_EFFORT: Optional[str] = "high"  # e.g. "high", "minimal", or None

# Batching
BATCH_SIZE: int = 10

# Common input files (usually unchanged)
PROMPT_FILE: str = "coding_prompt_2.txt"
PAPERS_FILE: str = "papers_original.txt"
MASTER_SHEET_SHUFFLED: str = "shuffled_master_sheet.xlsx"
MASTER_SHEET_FULL: str = "master-sheet-full.xlsx"

# Optional: cost calculation input
PRICING_FILE: str = "preise_modelle_LLM.xlsx"


# =========================
# Derived helpers (do not edit)
# =========================

def _safe_model_name(model: str) -> str:
    # safe for Windows/Linux paths
    return model.replace(":", "_").replace("/", "_").replace("\\", "_")


def _temp_tag(temp: float) -> str:
    # 1.0 -> "temp1", 0.7 -> "temp0.7"
    if float(temp).is_integer():
        return f"temp{int(temp)}"
    return f"temp{temp}"


def version_tag(
    experiment_id: str = EXPERIMENT_ID,
    run_variant: str = RUN_VARIANT,
    temperature: Optional[float] = TEMPERATURE,
    reasoning_effort: Optional[str] = REASONING_EFFORT,
) -> str:
    """
    Builds the 'version' part used in filenames/folders, e.g.:
      - "2-high-reasoning"
      - "2-temp1"
      - "2b-temp1"
      - "8-minimal-reasoning"
      - "3"
    """
    tag = f"{experiment_id}{run_variant}".strip()

    parts = [tag] if tag else []

    if temperature is not None:
        parts.append(_temp_tag(temperature))

    if reasoning_effort is not None and str(reasoning_effort).strip():
        effort = str(reasoning_effort).strip().replace(" ", "-")
        parts.append(f"{effort}-reasoning")

    return "-".join(parts)


def run_slug(
    experiment_id: str = EXPERIMENT_ID,
    run_variant: str = RUN_VARIANT,
    model: str = MODEL,
    temperature: Optional[float] = TEMPERATURE,
    reasoning_effort: Optional[str] = REASONING_EFFORT,
    batch_size: int = BATCH_SIZE,
) -> str:
    vtag = version_tag(
        experiment_id=experiment_id,
        run_variant=run_variant,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
    safe_model = _safe_model_name(model)
    return f"{vtag}_{safe_model}_bs-{batch_size}"


def outputs_dir(**overrides) -> Path:
    return Path(f"outputs_{run_slug(**overrides)}")


def responses_dir(**overrides) -> Path:
    return Path(f"responses_{run_slug(**overrides)}")


def matched_sheet_path(full: bool = False, suffix: str = "", **overrides) -> Path:
    """
    full=False => matched_master_sheet_<slug>.xlsx
    full=True  => matched_master_sheet_full_<slug>.xlsx

    suffix examples:
      - "" (default)
      - "_with_positions"
    """
    base = "matched_master_sheet_full" if full else "matched_master_sheet"
    return Path("matched_sheets") / f"{base}_{run_slug(**overrides)}{suffix}.xlsx"


def cost_duration_output_path(**overrides) -> Path:
    return Path("cost_duration") / f"cost_duration_{run_slug(**overrides)}.xlsx"


@dataclass(frozen=True)
class ExperimentSettings:
    experiment_id: str = EXPERIMENT_ID
    run_variant: str = RUN_VARIANT
    model: str = MODEL
    temperature: Optional[float] = TEMPERATURE
    reasoning_effort: Optional[str] = REASONING_EFFORT
    batch_size: int = BATCH_SIZE

    @property
    def safe_model(self) -> str:
        return _safe_model_name(self.model)

    @property
    def version(self) -> str:
        return version_tag(
            experiment_id=self.experiment_id,
            run_variant=self.run_variant,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )

    @property
    def slug(self) -> str:
        return run_slug(
            experiment_id=self.experiment_id,
            run_variant=self.run_variant,
            model=self.model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            batch_size=self.batch_size,
        )

    def outputs_dir(self) -> Path:
        return outputs_dir(
            experiment_id=self.experiment_id,
            run_variant=self.run_variant,
            model=self.model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            batch_size=self.batch_size,
        )

    def responses_dir(self) -> Path:
        return responses_dir(
            experiment_id=self.experiment_id,
            run_variant=self.run_variant,
            model=self.model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            batch_size=self.batch_size,
        )

    def matched_sheet_path(self, full: bool = False, suffix: str = "") -> Path:
        return matched_sheet_path(
            full=full,
            suffix=suffix,
            experiment_id=self.experiment_id,
            run_variant=self.run_variant,
            model=self.model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            batch_size=self.batch_size,
        )

    def cost_duration_output_path(self) -> Path:
        return cost_duration_output_path(
            experiment_id=self.experiment_id,
            run_variant=self.run_variant,
            model=self.model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            batch_size=self.batch_size,
        )

    def with_overrides(self, **kwargs) -> "ExperimentSettings":
        """
        Convenience for scripts that compare multiple runs (e.g. run_variant="b"/"c").
        """
        return replace(self, **kwargs)


SETTINGS = ExperimentSettings()
