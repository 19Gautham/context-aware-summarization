import json
import time
import re
from google import genai

API_KEY = "AIzaSyC4FALIVMPjoJfaeON4229CNyrAQY_lFVM"

client = genai.Client(api_key=API_KEY)
model = "gemini-2.0-flash"

def make_prompt(batch):
    qa = "\n\n".join([
        f"""Q: {e['question']}
Generated Answer: {e['ans']}
Evidence: {e['gold_evidence']}
Ground Truth: {e['ground_truth']}""" for e in batch
    ])

    prompt = f"""
Evaluate each generated answer compared to the ground truth, taking into account the provided evidence.

Use the following rubric:

- 1.0: Fully correct. Answer matches ground truth exactly (facts, numbers, entities, critical details).
- 0.9: Minor differences (e.g., slight rewording, approximate numbers within 2% margin).
- 0.8: Semantically very close, but missing some minor points or wording differences.
- 0.6: Partially correct. Covers some aspects but misses important parts or gives incomplete answer.
- 0.3: Major factual mistakes, missing critical information.
- 0.0: Completely incorrect, irrelevant, or contradicts ground truth.

When evaluating:

- Use the Evidence to verify if the generated answer can be justified even if the wording differs.
- If the Answer contradicts the Evidence or misses key facts from it, penalize accordingly.
- Minor differences in phrasing that preserve meaning are acceptable at 0.9–1.0.
- Pay special attention to numbers, counts, and critical named entities.

Important Instructions:

- Return the scores strictly in the same order as the questions are presented.
- Do not shuffle or reorder the answers.
- Ensure that the first score corresponds to the first question, the second score to the second question, and so on.
- Return only a plain JSON list of numbers (e.g., [1.0, 0.8, 0.6, 0.0]).
- Do not add any explanation, commentary, or any text.

Now, here are the questions, answers, and evidence:

{qa}

Scores:
""".strip()

    return prompt


def extract_json_list(response):
    # Safely extract the real text part
    try:
        text = response.candidates[0].content.parts[0].text
    except Exception as e:
        raise ValueError(f"Failed to extract text from response structure: {e}")

    # Clean up triple backticks if present
    text = re.sub(r"^```json\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text.strip())

    return json.loads(text)


# ---------- Call Gemini and Retry ----------
def get_scores_from_gemini(batch, max_retries=3):
    prompt = make_prompt(batch)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            scores = extract_json_list(response)

            if not isinstance(scores, list):
                raise ValueError("Parsed scores is not a list.")
            if len(scores) != len(batch):
                raise ValueError(f"Mismatch: expected {len(batch)} scores, got {len(scores)}.")
            return scores

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 10
                print(f"Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Padding with None.")
                return [None] * len(batch)

# ---------- Main Retry + Save Script ----------
INPUT_FILE = "output_mmqa_text_scored-fin_RETRY.json"
OUTPUT_FILE = "output_mmqa_text_scored-fin_RETRY2.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

entries_done = []
entries_to_retry = []

for entry in data:
    if entry.get("factual_score") is None:
        entries_to_retry.append(entry)
    else:
        entries_done.append(entry)

print(f"Entries to retry: {len(entries_to_retry)}")
print(f"Entries already fine: {len(entries_done)}")

# Retry in batches
batch_size = 3
start_time = time.time()

for i in range(0, len(entries_to_retry), batch_size):
    batch = entries_to_retry[i:i + batch_size]
    print(f"Processing retry batch {i // batch_size + 1}...")

    scores = get_scores_from_gemini(batch)

    for entry, score in zip(batch, scores):
        entry["factual_score"] = score

    entries_done.extend(batch)

    # Respect rate limit
    time.sleep(3)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(entries_done, f, indent=2, ensure_ascii=False)

elapsed = time.time() - start_time
print(f"\nRetried and scored {len(entries_to_retry)} entries in {elapsed:.2f} seconds.")
print(f"All results saved to {OUTPUT_FILE}")
