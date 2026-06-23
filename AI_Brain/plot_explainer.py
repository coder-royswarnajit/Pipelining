import ast
import json

from AI_Brain.llm_client import ask_llm
from AI_Brain.prompt_templates import plot_explanation_prompt


BATCH_SIZE = 5


def _clean_llm_json_response(response):
    cleaned_response = (
        str(response)
        .replace("```json", "")
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    try:
        return json.loads(cleaned_response)
    except Exception:
        try:
            return ast.literal_eval(cleaned_response)
        except Exception:
            return {}


def _compact_plot_payload(plot_payload):
    compact_payload = []

    for item in plot_payload or []:
        if not isinstance(item, dict):
            continue

        compact_item = {
            "plot_key": item.get("plot_key") or item.get("plot_name"),
            "summary": item.get("summary") or item.get("title") or "",
        }

        if item.get("plot_type"):
            compact_item["plot_type"] = item.get("plot_type")

        if item.get("insights"):
            compact_item["statistics"] = item.get("insights")

        compact_payload.append(compact_item)

    return compact_payload


def explain_all_plots(plot_payload):

    all_explanations = {}

    compact_payload = _compact_plot_payload(plot_payload)

    for start_idx in range(0, len(compact_payload), BATCH_SIZE):
        batch = compact_payload[start_idx:start_idx + BATCH_SIZE]

        print(f"Processing batch {start_idx // BATCH_SIZE + 1}")
        print("Batch plots:", len(batch))

        prompt = plot_explanation_prompt(plot_payload=batch)

        response = ask_llm(prompt)
        batch_results = _clean_llm_json_response(response)
        print(type(batch_results))

        if isinstance(batch_results, list):
            for item in batch_results:
                if isinstance(item, dict):
                    all_explanations.update(item)

        elif isinstance(batch_results, dict):
            all_explanations.update(batch_results)

        else:
            print(f"Failed batch {start_idx // BATCH_SIZE + 1}")
            print(batch_results)

    return all_explanations