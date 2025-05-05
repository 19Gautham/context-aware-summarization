import json
import time
import re
from google import genai

API_KEY = "AIzaSyC4FALIVMPjoJfaeON4229CNyrAQY_lFVM"
client = genai.Client(api_key=API_KEY)
model = "gemini-2.0-flash"

def make_prompt(batch):
    qa = "\n\n".join([
        f"Q: {e['question']}\nA: {e['answer']}\nGT: {e['ground_truth']}" for e in batch
    ])
    prompt = f"""
Your task is to evaluate each answer compared to its provided ground truth.

Use the following rubric to assign a factual correctness score between 0 and 1:

- 1.0: Exact match with ground truth (including correct numbers, names, details, ignore minor errors like missing unit/currency notation).
- 0.9: Numbers match approximately (within 2%), or small minor differences in phrasing but identical meaning.
- 0.8: Semantically very close, but some minor missing information.
- 0.6: Partially correct. Covers some correct aspects but missing or slightly wrong elsewhere.
- 0.3: Incomplete or partially wrong. Major missing elements.
- 0.0: Completely incorrect, contradictory, or nonsensical.

When evaluating numbers, small differences (like 110,389 vs 110389) should be treated carefully — if they are within 2% relative error, treat them as approximately matching.

Return only a JSON list of scores in order, like: [0.9, 0.7, 1.0, ...]

Now, here are the questions and answers:

{qa}

Scores:
""".strip()
    return prompt


def extract_json_list(text):
    text = re.sub(r"^```json\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text.strip())
    return json.loads(text)

def get_scores_from_gemini(batch, max_retries=3):
    prompt = make_prompt(batch)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            scores = extract_json_list(response.text)
            if not isinstance(scores, list) or len(scores) != len(batch):
                raise ValueError("Invalid score format or length mismatch.")
            return scores
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 10
                print(f"🔁 Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("❌ Max retries reached. Padding with None.")
                return [None] * len(batch)

# ----------------------- Main Retry Script -----------------------
INPUT_FILE = "output_mmqa_summary_scored-fixed2.json"
OUTPUT_FILE = "output_mmqa_summary_scored_retry5.json"
BATCH_SIZE = 5

# Load data (JSONL or JSON)
data = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

entries_done = []
entries_to_retry = []

for entry in data:
    if entry.get("factual_score") is None:
        ans = (entry.get("answer") or "").strip().lower()
        if ans == "":
            entry["factual_score"] = 0.0
            entries_done.append(entry)
        else:
            entries_to_retry.append(entry)
    else:
        entries_done.append(entry)

print(f"Entries to retry: {len(entries_to_retry)}")
print(f"Already valid/skipped entries: {len(entries_done)}")

# Retry in batches
start_time = time.time()

for i in range(0, len(entries_to_retry), BATCH_SIZE):
    batch = entries_to_retry[i:i + BATCH_SIZE]
    print(f"📦 Retrying batch {i // BATCH_SIZE + 1}/{(len(entries_to_retry) + BATCH_SIZE - 1) // BATCH_SIZE}")

    scores = get_scores_from_gemini(batch)
    for entry, score in zip(batch, scores):
        entry["factual_score"] = score

    entries_done.extend(batch)
    time.sleep(4)

# Write all entries back to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in entries_done:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

elapsed = time.time() - start_time
print(f"\nRetried and saved {len(entries_to_retry)} entries in {elapsed:.2f}s.")
print(f"Final output saved to: {OUTPUT_FILE}")