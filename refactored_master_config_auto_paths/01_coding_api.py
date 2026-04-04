from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
import asyncio

# version 04/03/2026

from master_config import SETTINGS, PROMPT_FILE, PAPERS_FILE  # central experiment settings


load_dotenv()
client = AsyncOpenAI(api_key=os.environ.get("UNC"))

# Step 1: Load the prompt
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt_text = f.read().strip()

# Step 2: Load the papers
with open(PAPERS_FILE, "r", encoding="utf-8") as f:
    papers_text = f.read().strip()


def build_prompt_with_papers(prompt, papers_text, start_idx=0, end_idx=5):
    """
    Combine prompt with a flexible range of papers.

    start_idx: index of first paper (inclusive, 0-based)
    end_idx: index of last paper (exclusive)
    """
    paper_blocks = papers_text.strip().split("\n\n")
    selected_papers = paper_blocks[start_idx:end_idx]
    combined_text = prompt + "\n\n" + "\n\n".join(selected_papers)
    return combined_text


CONCURRENT_REQUESTS = 100
START = 0
END = 5000
VERSION = SETTINGS.version
BATCH_SIZE = SETTINGS.batch_size
MODEL = SETTINGS.model

OUTPUTS_DIR = SETTINGS.outputs_dir()
RESPONSES_DIR = SETTINGS.responses_dir()


async def run_task(start, end, input, sem):
    result = {
        "start": start,
        "end": end,
        "output": "",
        "success": True,
        "exception": None,
    }

    try:
        async with sem:
            request_kwargs = {'model': MODEL, 'input': input}
            if SETTINGS.reasoning_effort is not None:
                request_kwargs['reasoning'] = {'effort': SETTINGS.reasoning_effort}
            if SETTINGS.temperature is not None:
                request_kwargs['temperature'] = SETTINGS.temperature
            response = await client.responses.create(**request_kwargs)
    except Exception as e:
        print(f"❌ request failed for batch {start+1}-{end}: {e}")
        result["exception"] = e
        result["success"] = False
        return result

    print(f"=== LLM Output for papers {start+1} to {end} ===")
    print(response)

    try:
        with open(
            OUTPUTS_DIR / f"{start+1}_{end}_coding_output_{VERSION}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(response.output_text)

        with open(
            RESPONSES_DIR / f"{start+1}_{end}_coding_response_{VERSION}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(str(response))
    except Exception as e:
        print(f"❌ creating files failed for batch {start+1}-{end}: {e}")
        result["exception"] = e
        result["success"] = False
        return result

    result["output"] = response.output_text
    return result


async def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

    tasks = []
    for i in range(START, END, BATCH_SIZE):
        batch_start = i
        batch_end = min(i + BATCH_SIZE, END)
        test_input = build_prompt_with_papers(
            prompt_text, papers_text, start_idx=batch_start, end_idx=batch_end
        )
        tasks.append(run_task(batch_start, batch_end, test_input, sem))

    results = await asyncio.gather(*tasks)
    failures = [r for r in results if not r["success"]]

    for r in failures:
        print(
            f"  -> failed to process batch {r['start']} - {r['end']}: {r['exception']}"
        )

    print(
        f"processed {len(results)} tasks, {len(results) - len(failures)} successful, {len(failures)} failures"
    )


asyncio.run(main())