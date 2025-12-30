from dotenv import load_dotenv
import os
from openai import OpenAI
import json

load_dotenv()
client = OpenAI(
      api_key=os.environ.get("UNC")
)

# Step 1: Load the prompt
with open("coding_prompt.txt", "r", encoding="utf-8") as f:
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

start = 5
end = 10 
# Example usage: papers 1 to 5 (0-based index)
test_input = build_prompt_with_papers(prompt_text, papers_text, start_idx=start, end_idx=end)
print(test_input)

# Send to the model
response = client.responses.create(
    model="gpt-4.1-mini",  # small GPT-4 variant
    input=test_input,
    temperature=0
)

print("=== LLM Output ===")
print(response.output_text)
print(response)

with open(f".\\outputs\\{start}_{end}_coding_output.txt", "w", encoding="utf-8") as f:
    f.write(response.output_text)

with open(f".\\responses\\{start}_{end}_coding_response.txt", "w", encoding="utf-8") as f:
    f.write(str(response))