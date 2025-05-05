import json

input_path = "output_mmqa_summary_scored_retry5.json"
output_path = "output_mmqa_summary_scored-fin10.json"

data = []

with open(input_path, "r", encoding="utf-8") as infile:
    for line in infile:
        line = line.strip().rstrip(",")  # Remove trailing comma if any
        if not line:
            continue
        try:
            obj = json.loads(line)
            data.append(obj)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid line:\n{line[:80]}...\nError: {e}")

# Write the list of JSON objects to a proper JSON array
with open(output_path, "w", encoding="utf-8") as outfile:
    json.dump(data, outfile, indent=2, ensure_ascii=False)
