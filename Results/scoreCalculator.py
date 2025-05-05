import json

# Path to results file
INPUT_FILE = "output_mmqa_summary_scored-fin10.json"

# Load all entries
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize counters
count_total = 0
count_1_0 = 0
count_0_9 = 0
count_0_8 = 0
count_0_6 = 0
count_0_3 = 0
count_rest = 0
sum_scores = 0.0

# Process each entry
for entry in data:
    score = entry.get("factual_score")
    if score is None:
        print("Error")
        continue  # Skip if still missing (shouldn't happen ideally)

    if isinstance(score, list):
        score = score[0]
    elif isinstance(score, str):
        score = float(score)
    # elif isinstance(score, dict):
    #     print(score)
    #     print(entry["uid"])
    #     score = score["score"]

    count_total += 1
    sum_scores += score


    if score == 0.3:
        count_0_3 += 1
    elif score == 0.6:
        count_0_6 += 1
    elif score == 0.8:
        count_0_8 += 1
    elif score == 0.9:
        count_0_9 += 1
    elif score == 1.0:
        count_1_0 += 1
    else:
        count_rest += 1

# Final stats
average_score = sum_scores / count_total if count_total > 0 else 0.0

print(f"Processed {count_total} entries.\n\n")
print(f"Scores >= 1.0 : {count_1_0}")
print(f"Scores >= 0.9 : {count_0_9 + count_1_0}")
print(f"Scores >= 0.8 : {count_0_8 + count_0_9 + count_1_0}")
print(f"Scores >= 0.6 : {count_0_6 + count_0_8 + count_0_9 + count_1_0}")
print(f"Scores >= 0.3 : {count_0_3 + count_0_6 + count_0_8 + count_0_9 + count_1_0}")
print(f"Scores rest : {count_rest}")
print(f"Average Score : {average_score:.4f}")

print(f"\n\nAccuracy Score >= 1.0 : {(count_1_0 )/ count_total}")
print(f"Accuracy Score >= 0.9 : {(count_0_9 + count_1_0)/ count_total}")
print(f"Accuracy Score >= 0.8 : {(count_0_8 + count_0_9 + count_1_0)/ count_total}")
print(f"Accuracy Score >= 0.6 : {(count_0_6 + count_0_8 + count_0_9 + count_1_0)/ count_total}")
print(f"Accuracy Score >= 0.3 : {(count_0_3 + count_0_6 + count_0_8 + count_0_9 + count_1_0)/ count_total}")
print(f"Accuracy Score == 0.0 : {count_rest/ count_total}")

# split
print(f"\n\nScore split")

print(f"Scores = 1.0 : {count_1_0}")
print(f"Scores = 0.9 : {count_0_9}")
print(f"Scores = 0.8 : {count_0_8}")
print(f"Scores = 0.6 : {count_0_6}")
print(f"Scores = 0.3 : {count_0_3}")
print(f"Scores rest : {count_rest}")
