import os
import requests
import json

# CONFIG
START = 0
END = 1000
VERSION = 5
BATCH_SIZE = 1
MODEL = "ministral-3:8b"
SAFE_MODEL = MODEL.replace(":", "_")

# Load prompt
with open("coding_prompt_5.txt", "r", encoding="utf-8") as f:
    prompt_text = f.read().strip()

# Load papers
with open("papers_original.txt", "r", encoding="utf-8") as f:
    papers_text = f.read().strip()


def build_prompt_with_papers(prompt, papers_text, start_idx=0, end_idx=5):
    paper_blocks = papers_text.strip().split("\n\n")
    selected_papers = paper_blocks[start_idx:end_idx]
    return prompt + "\n\n" + "\n\n".join(selected_papers)


def call_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
            "temperature": 0.0
            }
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def main():
    os.makedirs(f"outputs_{VERSION}_{SAFE_MODEL}_bs-{BATCH_SIZE}", exist_ok=True)
    os.makedirs(f"responses_{VERSION}_{SAFE_MODEL}_bs-{BATCH_SIZE}", exist_ok=True)

    for i in range(START, END, BATCH_SIZE):
        batch_start = i
        batch_end = min(i + BATCH_SIZE, END)

        full_prompt = build_prompt_with_papers(
            prompt_text, papers_text,
            start_idx=batch_start,
            end_idx=batch_end
        )

        print(f"Processing papers {batch_start + 1}–{batch_end}")

        try:
            ollama_json = call_ollama(full_prompt)
            output_text = ollama_json["response"]
        except Exception as e:
            print(f"❌ failed for batch {batch_start + 1}-{batch_end}: {e}")
            continue

        with open(
            f"./outputs_{VERSION}_{SAFE_MODEL}_bs-{BATCH_SIZE}/{batch_start + 1}_{batch_end}_coding_output_{VERSION}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(output_text)

        with open(
            f"./responses_{VERSION}_{SAFE_MODEL}_bs-{BATCH_SIZE}/{batch_start + 1}_{batch_end}_coding_response_{VERSION}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(ollama_json, f, indent=2)

    print("✅ Done")


if __name__ == "__main__":
    main()
