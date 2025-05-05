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

# I/O files
INPUT_FILE = "../../../output/mmqa_summarization_agent.json"
OUTPUT_FILE = "output_mmqa_summary_scored.jsonl"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

write_buffer = []
start_time = time.time()

for i in range(0, len(data), 25):
    batch = data[i:i+25]
    print(f"📦 Processing batch {i//25 + 1}...")

    scores = get_scores_from_gemini(batch)
    for entry, score in zip(batch, scores):
        entry["factual_score"] = score
        write_buffer.append(entry)

    if len(write_buffer) >= 125:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for item in write_buffer:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ Wrote {len(write_buffer)} entries to file.")
        write_buffer.clear()

    time.sleep(5)  # Optional rate limit

# Final flush
if write_buffer:
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for item in write_buffer:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Final write: {len(write_buffer)} entries flushed.")

elapsed = time.time() - start_time
print(f"\n✅ Scored {len(data)} entries in {elapsed:.2f} seconds.")
print(f"📄 Output saved to {OUTPUT_FILE}")
