import re
from pathlib import Path
import pandas as pd

BATCH_SIZE = 1
MODEL_NAME = "qwen3:8b"
SAFE_MODEL = MODEL_NAME.replace(":", "_")

TXT_DIR = Path(f"responses_2_{SAFE_MODEL}_bs-{BATCH_SIZE}")
OUTPUT_DIR = Path("cost_duration")
OUTPUT_FILE = OUTPUT_DIR / f"duration_only_{SAFE_MODEL}_bs-{BATCH_SIZE}.xlsx"

OUTPUT_DIR.mkdir(exist_ok=True)

# Ollama-Zeitfelder (Nanosekunden)
PATTERNS = {
    "total_duration": re.compile(r'"total_duration":\s*(\d+)'),
    "load_duration": re.compile(r'"load_duration":\s*(\d+)'),
    "prompt_eval_duration": re.compile(r'"prompt_eval_duration":\s*(\d+)'),
    "eval_duration": re.compile(r'"eval_duration":\s*(\d+)'),
}

rows = []

for txt_file in TXT_DIR.glob("*.txt"):
    text = txt_file.read_text(encoding="utf-8", errors="ignore")

    extracted = {}
    for key, pattern in PATTERNS.items():
        match = pattern.search(text)
        if not match:
            print(f"⚠️ {key} fehlt in {txt_file.name}")
            break
        extracted[key] = int(match.group(1))
    else:
        # Nanosekunden → Sekunden
        total_sec = extracted["total_duration"] / 1_000_000_000
        inference_sec = (extracted["prompt_eval_duration"] + extracted["eval_duration"]) / 1_000_000_000

        rows.append({
            "file": txt_file.name,
            "model": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "total_sec": total_sec,
            "inference_sec": inference_sec,
            "per_paper_total_sec": total_sec / BATCH_SIZE,
            "per_paper_inference_sec": inference_sec / BATCH_SIZE,
        })

df = pd.DataFrame(rows)

# Zusammenfassende Metriken
summary_metrics = {
    "avg_total_per_file_sec": df["total_sec"].mean(),
    "avg_inference_per_file_sec": df["inference_sec"].mean(),
    "avg_total_per_paper_sec": df["per_paper_total_sec"].mean(),
    "avg_inference_per_paper_sec": df["per_paper_inference_sec"].mean(),
    "total_total_sec": df["total_sec"].sum(),
    "total_inference_sec": df["inference_sec"].sum(),
}

summary_df = pd.DataFrame(
    list(summary_metrics.items()),
    columns=["metric", "value"]
)

# Ausgabe
print("\n=== ZEIT ===")
print(f"Ø total pro File:       {summary_metrics['avg_total_per_file_sec']:.2f} s")
print(f"Ø inference pro File:   {summary_metrics['avg_inference_per_file_sec']:.2f} s")
print(f"Ø total pro Paper:      {summary_metrics['avg_total_per_paper_sec']:.2f} s")
print(f"Ø inference pro Paper:  {summary_metrics['avg_inference_per_paper_sec']:.2f} s")
print(f"Gesamtdauer total:      {summary_metrics['total_total_sec']:.2f} s")
print(f"Gesamtdauer inference:  {summary_metrics['total_inference_sec']:.2f} s")

# Excel schreiben
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="details", index=False)
    summary_df.to_excel(writer, sheet_name="summary", index=False)

print(f"\n✅ Excel-Datei erstellt: {OUTPUT_FILE}")
