"""Workflow Mode Detection and Configuration.

Detects whether a dataset is single-discipline or multi-discipline
based on user instruction keywords.

Single-discipline workflow:  rough → base_model → fine
Multi-discipline workflow:   discipline_discovery → rough → base_model → fine
"""

import re


class WorkflowMode:
    SINGLE_DISCIPLINE_QUALITY = "single_discipline_quality"
    MULTI_DISCIPLINE_SUBJECT = "multi_discipline_subject"


# ── Detection ────────────────────────────────────────────────────

_MULTI_INSTRUCTION_PATTERNS = [
    r"多学科", r"multi[- ]?discipline", r"multiple\s+subject",
    r"cross[- ]?domain", r"各(个)?学科", r"分.*学科",
    r"多(个|种).*领域", r"multi[- ]?domain"
]


def detect_workflow_mode(user_instruction: str) -> str:
    """Detect workflow mode from user instruction keywords."""
    instruction_lower = user_instruction.lower()
    for pattern in _MULTI_INSTRUCTION_PATTERNS:
        if re.search(pattern, instruction_lower):
            return WorkflowMode.MULTI_DISCIPLINE_SUBJECT
    return WorkflowMode.SINGLE_DISCIPLINE_QUALITY


# ── Workflow ordering ────────────────────────────────────────────

def get_workflow_order(mode: str) -> list:
    """Get the order of filtering stages based on mode.

    Single-discipline:  rough → base_model → fine
    Multi-discipline:   discipline_discovery → rough → base_model → fine
    """
    if mode == WorkflowMode.MULTI_DISCIPLINE_SUBJECT:
        return [
            "discipline_discovery_agent",
            "rough_filter_agent",
            "base_model_filter_agent",
            "fine_filter_agent",
        ]
    else:
        return [
            "rough_filter_agent",
            "base_model_filter_agent",
            "fine_filter_agent",
        ]


# ── Fine-filter dimensions ───────────────────────────────────────

def get_fine_filter_dimensions(mode: str, user_instruction: str = "") -> dict:
    """Get fine-filter dimensions based on workflow mode."""
    if mode == WorkflowMode.MULTI_DISCIPLINE_SUBJECT:
        subjects = extract_target_subjects(user_instruction)
        subject_list = ", ".join(subjects) if subjects else "target subject"

        return {
            "subject_relevance": {
                "description": f"Relevance to target subjects: {subject_list}",
                "weight": "float, set appropriate value"
            },
            "clarity": {
                "description": "Clarity and coherence of the content",
                "weight": "float, set appropriate value"
            },
            "quality": {
                "description": "Overall quality and educational value",
                "weight": "float, set appropriate value"
            }
        }
    else:
        return {
            "accuracy": {
                "description": "Factual correctness and precision",
                "weight": "float, set appropriate value"
            },
            "complexity": {
                "description": "Appropriate difficulty and depth",
                "weight": "float, set appropriate value"
            },
            "clarity": {
                "description": "Clear explanation and reasoning",
                "weight": "float, set appropriate value"
            },
            "educational_value": {
                "description": "Learning value and insight",
                "weight": "float, set appropriate value"
            }
        }


# ── Subject extraction ───────────────────────────────────────────

def extract_target_subjects(instruction: str) -> list:
    """Extract target subject names from instruction."""
    subjects = []
    subject_keywords = {
        "数学": "mathematics", "物理": "physics", "化学": "chemistry",
        "生物": "biology", "医学": "medicine", "医疗": "medicine",
        "金融": "finance", "编程": "programming", "代码": "programming",
        "法律": "law", "历史": "history",
        "math": "mathematics", "physics": "physics",
        "chemistry": "chemistry", "biology": "biology",
        "medicine": "medicine", "medical": "medicine",
        "finance": "finance", "code": "programming",
        "programming": "programming", "law": "law",
    }

    instruction_lower = instruction.lower()
    for keyword, subject in subject_keywords.items():
        if keyword in instruction_lower and subject not in subjects:
            subjects.append(subject)

    return subjects if subjects else ["target subject"]
