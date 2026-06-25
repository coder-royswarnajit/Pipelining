import json

from AI_Brain.llm_client import ask_llm
from AI_Brain.prompt_templates import build_metric_recommendation_prompt
from Modelling.evaluations import (
    get_default_ranking_metric,
    normalize_ranking_metric,
    resolve_ranking_metric,
)


def _build_target_context(df, target_column, problem_type):
    """Compact target statistics for the LLM prompt."""

    if target_column is None or target_column not in df.columns:
        return {}

    series = df[target_column].dropna()
    context = {
        "target_column": target_column,
        "problem_type": problem_type,
        "non_null_count": int(series.shape[0]),
        "missing_count": int(df[target_column].isnull().sum()),
        "unique_count": int(series.nunique()),
    }

    if problem_type == "classification":
        value_counts = series.value_counts(normalize=True).head(10)
        context["class_distribution"] = {
            str(label): round(float(pct) * 100, 2)
            for label, pct in value_counts.items()
        }
        if len(value_counts) > 0:
            context["majority_class_percent"] = round(
                float(value_counts.iloc[0]) * 100, 2
            )
    else:
        if len(series) > 0 and series.dtype.kind in "biufc":
            context["target_stats"] = {
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
            }

    return context


def _parse_metric_response(response):
    if not response:
        return {}

    cleaned = response.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    return {}


def _rule_based_fallback(metadata, df, target_column, problem_type):
    """
    Deterministic fallback when the LLM response is missing or invalid.
    """

    problem_type = (problem_type or "").lower()
    default_metric = get_default_ranking_metric(problem_type)

    if problem_type == "classification" and target_column in df.columns:
        series = df[target_column].dropna()
        if len(series) > 0:
            majority_pct = float(series.value_counts(normalize=True).iloc[0])
            if majority_pct >= 0.80:
                return {
                    "metric": "recall",
                    "reason": (
                        "Majority class exceeds 80%, so recall helps prioritize "
                        "performance on minority classes."
                    ),
                    "source": "rule_based_fallback",
                }
            if series.nunique() == 2:
                return {
                    "metric": "f1_score",
                    "reason": (
                        "Binary classification with moderate balance; F1 balances "
                        "precision and recall."
                    ),
                    "source": "rule_based_fallback",
                }

    if problem_type == "regression" and target_column in df.columns:
        series = df[target_column].dropna()
        if len(series) > 0 and series.std() > 0:
            cv = float(series.std() / abs(series.mean())) if series.mean() != 0 else 0
            if cv > 0.5:
                return {
                    "metric": "mae",
                    "reason": (
                        "Target shows high relative spread; MAE is robust and "
                        "easier to interpret than squared-error metrics."
                    ),
                    "source": "rule_based_fallback",
                }

    return {
        "metric": default_metric,
        "reason": f"Using default ranking metric for {problem_type}.",
        "source": "rule_based_fallback",
    }


def recommend_ranking_metric(metadata, df, target_column=None, problem_type=None):
    """
    Uses metadata and sample rows to recommend the primary evaluation metric
    for ranking models.

    Returns:
        {
            "metric": str,
            "reason": str,
            "higher_is_better": bool,
            "source": "ai" | "rule_based_fallback"
        }
    """

    problem_type = (problem_type or "classification").lower()
    sample_rows = df.head(5).to_dict(orient="records")
    target_context = _build_target_context(df, target_column, problem_type)

    try:
        prompt = build_metric_recommendation_prompt(
            metadata=metadata,
            sample_rows=sample_rows,
            target_column=target_column,
            problem_type=problem_type,
            target_context=target_context,
        )
        response = ask_llm(prompt)
        parsed = _parse_metric_response(response)

        metric = normalize_ranking_metric(
            parsed.get("metric"),
            problem_type,
        )

        if metric:
            resolved = resolve_ranking_metric(metric, problem_type)
            reason = str(parsed.get("reason", "")).strip()
            if not reason:
                reason = f"Recommended {metric} as the primary ranking metric."

            return {
                **resolved,
                "reason": reason,
                "source": "ai",
            }

    except Exception as e:
        print(f"Failed to generate metric recommendation: {e}")

    fallback = _rule_based_fallback(metadata, df, target_column, problem_type)
    resolved = resolve_ranking_metric(fallback["metric"], problem_type)
    return {
        **resolved,
        "reason": fallback["reason"],
        "source": fallback["source"],
    }
