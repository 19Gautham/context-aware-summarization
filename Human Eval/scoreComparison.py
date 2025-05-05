import os
import json
from glob import glob
from typing import Dict
from statistics import mean


def load_llm_scores(llm_file: str) -> Dict[str, float]:
    """Load LLM scores and return a map from UID to factual_score."""
    with open(llm_file, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)
    return {entry['uid']: entry['factual_score'] for entry in llm_data}


def load_human_scores(human_dir: str) -> Dict[str, float]:
    """Load all human eval files and return a map from UID to human_score."""
    human_scores = {}
    for file in glob(os.path.join(human_dir, "*.json")):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                uid = entry.get("uid")
                score = entry.get("human_score")
                if uid is not None and score is not None:
                    human_scores[uid] = score
    return human_scores


def evaluate(llm_map: Dict[str, float], human_map: Dict[str, float]) -> None:
    """Compare LLM scores to human scores and print evaluation metrics."""
    mae_list = []
    tolerance_01 = 0
    tolerance_02 = 0
    tolerance_03 = 0
    exact_match = 0
    total = 0

    error = 0

    for uid, human_score in human_map.items():
        llm_score = llm_map.get(uid)
        if llm_score is None:
            error += 1
            continue  # skip if LLM didn't score this UID

        diff = abs(human_score - llm_score)
        mae_list.append(diff)

        if diff <= 0.1:
            tolerance_01 += 1
        if diff <= 0.2:
            tolerance_02 += 1
        if diff <= 0.3:
            tolerance_03 += 1
        if human_score == llm_score:
            exact_match += 1

        total += 1

    mae = mean(mae_list) if mae_list else float('nan')

    print(f"Total comparisons: {total}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Agreement within 0.1 tolerance: {(tolerance_01 / total) * 100:.2f}%")
    print(f"Agreement within 0.2 tolerance: {(tolerance_02 / total) * 100:.2f}%")
    print(f"Agreement within 0.3 tolerance: {(tolerance_03 / total) * 100:.2f}%")
    print(f"Exact match: {(exact_match / total) * 100:.2f}%")

if __name__ == "__main__":
    LLM_FILE = "output_summary_gemini_scored-fin.json"
    HUMAN_DIR = "human_eval_folder/"

    llm_scores = load_llm_scores(LLM_FILE)
    human_scores = load_human_scores(HUMAN_DIR)
    evaluate(llm_scores, human_scores)
