import json

# Path to your file
INPUT_FILE = "output_mmqa_summary_scored-fin10.json"
OUTPUT_FILE = "output_mmqa_summary_scored-fixed2.json"

# Load all entries
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

count_fixed = 0

# Process each entry
for entry in data:
    score = entry.get("factual_score")
    if isinstance(score, dict):
        # entry["factual_score"] = None
        count_fixed += 1

print(f"✅ Fixed {count_fixed} entries with nested score dictionaries.")

# # Save back to new file (or overwrite INPUT_FILE if you prefer)
# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=2, ensure_ascii=False)
#
# print(f"📄 Cleaned results written to: {OUTPUT_FILE}")
