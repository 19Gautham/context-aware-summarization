import json
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_lg")

def checkSemanticSimilarityScore(ans, ref):
    return nlp(ans).similarity(nlp(ref))

def load_ground_truth(tatqa_path):

    with open(tatqa_path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    gt_map = {}
    for item in data:
        for q in item['questions']:
            uid = q['uid']
            answer = q['answer']
            if isinstance(answer, list):
                answer = ", ".join(map(str, answer))
            elif isinstance(answer, (int, float)):
                answer = str(answer)
            gt_map[uid] = answer

    return gt_map


def evaluate(pred_path, gt_path, out_path):

    # load the predictions from a file
    with open(pred_path, 'r', encoding="utf-8") as f:
        predictions = json.load(f)

    # let's load the map that contains data about the ground truth for each question
    groundTruth = load_ground_truth(gt_path)
    enriched = []

    for ex in tqdm(predictions, desc="Evaluating"):
        qid = ex.get("uid")
        ans = ex["answer"]

        ground_truth = groundTruth.get(qid, "")
        sim = checkSemanticSimilarityScore(ans, ground_truth)

        enriched.append({
            **ex,
            "ground_truth": ground_truth,
            "similarity": round(sim, 4)
            # "factuality": fact
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    print(f"\nSaved enriched results to: {out_path}")


if __name__ == "__main__":

    # predictions
    pred_file = "../../../agentData/gemma_1b_results.json"
    # ground truth
    gt_file = "../../../data/tatqa_dataset_train.json"
    # output
    out_file = "../../../output/tatqa_summary_agent.json"

    evaluate(pred_file, gt_file, out_file)
