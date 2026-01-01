from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
import asyncio

load_dotenv()
client = AsyncOpenAI(api_key=os.environ.get("UNC"))

# Step 1: Load the prompt
with open("coding_prompt_2.txt", "r", encoding="utf-8") as f:
    prompt_text = f.read().strip()

# Step 2: Load the papers
with open("papers_original.txt", "r", encoding="utf-8") as f:
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


CONCURRENT_REQUESTS = 10
START = 0
END = 100
VERSION = 1
BATCH_SIZE = 5


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
            response = await client.responses.create(
                model="gpt-4.1-mini", input=input, temperature=0  # small GPT-4 variant
            )
    except Exception as e:
        print(f"❌ request failed for batch {start+1}-{end}: {e}")
        result["exception"] = e
        result["success"] = False
        return result

    print(f"=== LLM Output for papers {start+1} to {end} ===")
    print(response)

    try:
        with open(
            f".\\outputs_{VERSION}\\{start+1}_{end}_coding_output_{VERSION}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(response.output_text)

        with open(
            f".\\responses_{VERSION}\\{start+1}_{end}_coding_response_{VERSION}.txt",
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
    os.makedirs(f"outputs_{VERSION}", exist_ok=True)
    os.makedirs(f"responses_{VERSION}", exist_ok=True)

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

    print(
        f"processed {len(results)} tasks, {len(results) - len(failures)} successful, {len(failures)} failures"
    )

    for r in failures:
        print(
            f"  -> failed to process batch {r['start']} - {r['end']}: {r['exception']}"
        )


asyncio.run(main())
