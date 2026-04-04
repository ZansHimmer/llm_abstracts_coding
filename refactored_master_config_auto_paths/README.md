# Refactor: master_config + derived paths

## What you change day-to-day
Edit **master_config.py** (top section "EDIT THESE"):

- EXPERIMENT_ID (e.g. "2", "8")
- RUN_VARIANT (e.g. "", "b", "c")
- MODEL
- TEMPERATURE (None, or 0.7, 1.0, ...)
- REASONING_EFFORT ("high", "minimal", or None)
- BATCH_SIZE

Everything else (output folders, response folders, matched sheet filenames, cost-duration filenames)
is derived automatically.

## Naming convention (auto)
A run gets a tag like:

  <version_tag>_<safe_model>_bs-<batch_size>

Examples:
- 2-high-reasoning_gpt-5-mini_bs-10
- 2b-temp1_qwen3_8b_bs-1
- 8-minimal-reasoning_gpt-5-mini_bs-1

and folders:
- outputs_<slug>/
- responses_<slug>/

matched sheets:
- matched_sheets/matched_master_sheet_<slug>.xlsx
- matched_sheets/matched_master_sheet_full_<slug>.xlsx

## Multiple-run overlap (03c)
03c_overlap_multiple_runs.py compares variants by default:
  RUN_VARIANTS = ["", "b", "c"]

It uses the same base settings as master_config.py, and only swaps RUN_VARIANT to build file paths.
