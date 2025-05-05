import json
import random

input_path = "output_summary_gemini_scored-fin.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

sampled_data = random.sample(data, 1000)

chunks = [sampled_data[i::6] for i in range(6)]

output_files = []
for idx, chunk in enumerate(chunks):
    for item in chunk:
        item["human_score"] = None  # placeholder
        item.pop("factual_score", None)  # remove model score
        item.pop("similarity", None)
    filename = f"splits/human_eval_group_{idx + 1}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(chunk, f, indent=2, ensure_ascii=False)
    output_files.append(filename)

print(output_files)