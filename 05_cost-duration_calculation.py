import re
from pathlib import Path
import pandas as pd

BATCH_SIZE = 5
MODEL_NAME = "gpt-4.1-mini"

TXT_DIR = Path(f"responses_2_{MODEL_NAME}_bs-{BATCH_SIZE}")
OUTPUT_DIR = Path("cost_duration")
OUTPUT_FILE = OUTPUT_DIR / f"cost_duration_2_{MODEL_NAME}_bs-{BATCH_SIZE}.xlsx"

PRICING_FILE = "preise_modelle_LLM.xlsx"

PATTERNS = {
    "created_at": re.compile(r"created_at=(\d+)"),
    "completed_at": re.compile(r"completed_at=(\d+)"),
    "input_tokens": re.compile(r"input_tokens=(\d+)"),
    "output_tokens": re.compile(r"output_tokens=(\d+)"),
}

OUTPUT_DIR.mkdir(exist_ok=True)

pricing = pd.read_excel(PRICING_FILE).set_index("model")

if MODEL_NAME not in pricing.index:
    raise ValueError(f"Kein Preiseintrag für Modell: {MODEL_NAME}")

PRICE_INPUT = pricing.loc[MODEL_NAME, "input_per_1m"]
PRICE_OUTPUT = pricing.loc[MODEL_NAME, "output_per_1m"]

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
        duration_sec = extracted["completed_at"] - extracted["created_at"]

        cost_input = extracted["input_tokens"] / 1_000_000 * PRICE_INPUT
        cost_output = extracted["output_tokens"] / 1_000_000 * PRICE_OUTPUT
        total_cost = cost_input + cost_output

        rows.append({
            "file": txt_file.name,
            "model": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "duration_sec": duration_sec,
            "duration_per_paper_sec": duration_sec / BATCH_SIZE,
            "input_tokens": extracted["input_tokens"],
            "output_tokens": extracted["output_tokens"],
            "cost_usd": total_cost,
            "cost_per_paper_usd": total_cost / BATCH_SIZE,
        })

df = pd.DataFrame(rows)

summary_metrics = {
    "avg_duration_per_file_sec": df["duration_sec"].mean(),
    "avg_duration_per_paper_sec": df["duration_per_paper_sec"].mean(),
    "total_duration_sec": df["duration_sec"].sum(),

    "avg_cost_per_file_usd": df["cost_usd"].mean(),
    "avg_cost_per_paper_usd": df["cost_per_paper_usd"].mean(),
    "total_cost_usd": df["cost_usd"].sum(),
}

summary_df = pd.DataFrame(
    list(summary_metrics.items()),
    columns=["metric", "value"]
)

print("\n=== ZEIT ===")
print(f"Ø Dauer pro File:   {summary_metrics['avg_duration_per_file_sec']:.2f} s")
print(f"Ø Dauer pro Paper:  {summary_metrics['avg_duration_per_paper_sec']:.2f} s")
print(f"Gesamtdauer:       {summary_metrics['total_duration_sec']:.2f} s")

print("\n=== KOSTEN ===")
print(f"Ø Kosten pro File:  ${summary_metrics['avg_cost_per_file_usd']:.6f}")
print(f"Ø Kosten pro Paper: ${summary_metrics['avg_cost_per_paper_usd']:.6f}")
print(f"Gesamtkosten:      ${summary_metrics['total_cost_usd']:.6f}")

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="details", index=False)
    summary_df.to_excel(writer, sheet_name="summary", index=False)

print(f"\n✅ Excel-Datei erstellt: {OUTPUT_FILE}")

