# Gemini QA Pipelines – TAT-QA and MMQA

This contains two Python-based QA extraction pipelines that use **Google Gemini 2.0 Flash** to generate concise **answers** and **supporting evidence** from large-scale question-answering datasets:

- **TAT-QA**: Text and Table-based Question Answering
- **MMQA**: Multi-hop Multimodal Question Answering

---

## Requirements

- Python 3.8 or higher
- [Google Generative AI Python SDK](https://github.com/google/generative-ai-python)

Install with:
```bash
pip install google-generativeai
```

---

## API Setup

Before running any script, configure the API:

```python
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")
```

---

## File Structure

| File Name              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `text_agent_tatqa.ipynb`    | Processes the TAT-QA dataset and generates answers & evidence using Gemini. |
| `text_agent_mmqa.ipynb`| Processes the MMQA dataset in batches and includes ground-truth comparison. |

---

## TAT-QA Pipeline

### Input:
- `tatqa_text_or_tabletext.json`

### Output:
- `tatqa_all_with_answers_and_evidence2.json`  
Each entry looks like:
```json
{
  "uid": "...",
  "question": "...",
  "ans": "...",
  "gold_evidence": "..."
}
```

### Features:
- Processes 3,600 questions from paragraph text.
- Calls Gemini for:
  - One-sentence answers (`Answer:`)
  - Natural language reasoning (`Evidence:`)
- Saves after every question (no batching).

### Run:
```bash
python text_agent_tatqa.ipynb
```

---

## MMQA Pipeline

### Input:
- `extracted_mmqa.json` with fields:
  - `qid`, `context`, `question`, `answer`

### Output:
- Batched files: `mmqa_batch_<start>_<end>.json`  
Each entry looks like:
```json
{
  "qid": "...",
  "question": "...",
  "ans": "...",
  "gold_evidence": "...",
  "gold_answer": "..."
}
```

### Features:
- Modular batch processing via `run_batch(start_idx, end_idx)`
- Stores both model answer and ground truth
- Exponential backoff for rate limit/quota handling
- Results saved once per batch

### Run:
Adjust and run in `text_agent_mmqa.ipynb`:
```python
batch_size = 500
start_from = 4000
end_at = 5000

for start in range(start_from, end_at, batch_size):
    run_batch(start, start + batch_size)
```

---

## Prompt Templates

### Answer Prompt:
```
Context:
{context}

Question:
{question}

Please provide a concise answer in exactly one sentence, without any further explanation or calculation steps. It should start with "Answer: "

Answer:
```

### Evidence Prompt:
```
In your evidence you should:
- Explain in full English why each piece of data supports your answer. Strictly start it with "Evidence:"

Context:
{context}

Question:
{question}

Answer and Evidence:
```

---

## Notes

- Gemini API errors like 429 (rate limit) or quota exceedance are handled with retries.
- MMQA code is more modular and suited for large-scale processing.
- TAT-QA code is better suited for smaller experimental runs.

---

## Contact

For questions, suggestions, or contributions, please open an issue or submit a pull request.

---