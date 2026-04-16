import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Step 0b: Generate structured SFT training data from Video-SafetyBench.

Produces train_llm/data/sft_data.jsonl — consumed by SFTDataset.

Fully deterministic — no model inference required. All fields are derived
from the dataset labels and scenario.json descriptions.

Output schema per refusal:
  {
    "action": "REFUSE" | "ESCALATE_WITH_ANALYSIS",
    "risk_category": str,
    "sub_category": str,
    "short_rationale": str
  }

Policy fields are always ground-truth correct:
  harmful → action=REFUSE
  benign  → action=ESCALATE_WITH_ANALYSIS

Run:
    python scripts/generate_sft_data.py
"""

import json
import os
import warnings

from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login

warnings.filterwarnings("ignore")
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "BAAI/Video-SafetyBench"
CATEGORIES = ("1_Violent_Crimes", "10_Hate", "12_Sexual_Content")
OUT_PATH   = Path("data/sft_data.jsonl")

SCENARIO_PATH = Path(__file__).resolve().parent.parent.parent / \
    "Video-SafetyBench" / "eval" / "scenario.json"

# Map dataset category → scenario.json key
CATEGORY_TO_SCENARIO_KEY = {
    "1_Violent_Crimes":  "Violent_Crimes",
    "10_Hate":           "Hate",
    "12_Sexual_Content": "Sexual_Content",
}

# Map dataset category → human-readable display name
CATEGORY_DISPLAY = {
    "1_Violent_Crimes":  "Violent Crimes",
    "10_Hate":           "Hate",
    "12_Sexual_Content": "Sexual Content",
}

# Deterministic policy by split
POLICY = {
    "harmful": "REFUSE",
    "benign":  "ESCALATE_WITH_ANALYSIS",
}

POLICY_LINE = {
    "harmful": (
        "This is an explicit harmful request. "
        'Set action="REFUSE".'
    ),
    "benign": (
        "This is a neutral request about a risky video. "
        'Set action="ESCALATE_WITH_ANALYSIS".'
    ),
}

QUESTION_TEMPLATE = """\
You are a content safety video moderator. Analyze the video and the user request below.
Output ONLY a valid JSON object with these exact fields — no explanation, no surrounding text:
{{
  "action": "REFUSE" or "ESCALATE_WITH_ANALYSIS",
  "risk_category": "<main category e.g. Violent Crimes, Hate, Sexual Content>",
  "sub_category": "<specific subcategory>",
  "short_rationale": "<concise moderation explanation>"
}}
{policy_line}

User request: {question}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_subcategory(raw: str) -> str:
    """'Animal_Abuse' → 'Animal Abuse'"""
    return raw.replace("_", " ")


def build_rationale(split: str, risk_category: str, sub_category: str,
                    harmful_intention: str, scenario_desc: str) -> str:
    if split == "harmful":
        return (
            f"The request involves {risk_category} content ({sub_category}). "
            f"{scenario_desc} "
            f"The request specifically targets: {harmful_intention}"
        )
    else:  # benign
        return (
            f"The video contains {risk_category} ({sub_category}) content. "
            f"{scenario_desc} "
            f"This neutral query about potentially harmful material is escalated for human review."
        )


def build_refusal(action: str, risk_category: str, sub_category: str,
                  short_rationale: str) -> str:
    return json.dumps({
        "action":          action,
        "risk_category":   risk_category,
        "sub_category":    sub_category,
        "short_rationale": short_rationale,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Load scenario descriptions
    with SCENARIO_PATH.open(encoding="utf-8") as f:
        scenario: dict[str, str] = json.load(f)
    print(f"Loaded scenario descriptions for: {list(scenario.keys())}")

    # Load dataset (both splits, filtered to target categories)
    print("Loading dataset...")
    items = []
    for split in ("harmful", "benign"):
        ds = load_dataset(DATASET_ID, split=split).filter(
            lambda x: x["category"] in CATEGORIES
        )
        for ex in ds:
            items.append({**ex, "split": split})
    print(f"  {len(items)} total items")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Start fresh — schema has changed
    written = 0

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for i, item in enumerate(items):
            qid        = item["question_id"]
            split      = item["split"]
            category   = item["category"]
            question   = item["question"]
            harmful_intention = item["harmful_intention"]
            raw_sub    = item.get("subcategory", "unknown")

            risk_category = CATEGORY_DISPLAY.get(category, category)
            sub_category  = normalize_subcategory(raw_sub)
            scenario_key  = CATEGORY_TO_SCENARIO_KEY.get(category, "")
            scenario_desc = scenario.get(scenario_key, "")
            action        = POLICY[split]

            short_rationale = build_rationale(
                split, risk_category, sub_category,
                harmful_intention, scenario_desc,
            )
            refusal_str = build_refusal(action, risk_category, sub_category, short_rationale)
            full_question = QUESTION_TEMPLATE.format(
                policy_line=POLICY_LINE[split],
                question=question,
            )

            record = {
                "question_id": qid,
                "question":    full_question,
                "video_path":  item["video_path"],
                "category":    category,
                "split":       split,
                "refusal":     refusal_str,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if (i + 1) % 50 == 0 or i < 5:
                print(f"[{i+1}/{len(items)}] {qid} | {split} | {action} | {risk_category} / {sub_category}")

    print(f"\nDone. Written={written} -> {OUT_PATH.resolve()}")

    # Quick sanity check
    with OUT_PATH.open(encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
    harmful_actions = {json.loads(d["refusal"])["action"] for d in lines if d["split"] == "harmful"}
    benign_actions  = {json.loads(d["refusal"])["action"] for d in lines if d["split"] == "benign"}
    print(f"harmful actions: {harmful_actions}")
    print(f"benign actions:  {benign_actions}")
    print(f"Sample refusal:\n{json.dumps(json.loads(lines[0]['refusal']), indent=2)}")


if __name__ == "__main__":
    main()
