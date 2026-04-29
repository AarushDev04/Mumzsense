# Escalation Handler: Gate B stub for Phase 1
# Handles urgent queries requiring human intervention
"""
MumzSense v1 — Escalation Handler
Gate B stub: typed return for urgent / low-confidence queries.
Phase 2 will replace this with a real paediatric triage agent.
"""
from __future__ import annotations
from typing import Dict, Any, Optional

# Static CTA messages (Phase 1)
ESCALATION_MESSAGES = {
    "en": {
        "title": "This sounds like something a healthcare professional should weigh in on.",
        "body": (
            "Based on what you've shared, I'd recommend speaking directly with your "
            "paediatrician or midwife rather than relying on community experiences alone. "
            "If you're concerned about an emergency, please call your local emergency services."
        ),
        "cta": "Find a paediatrician near you",
        "cta_url": "https://www.mumzworld.com/",  # Phase 2: real directory
    },
    "ar": {
        "title": "يبدو أن هذا يستدعي رأي متخصص في الرعاية الصحية.",
        "body": (
            "بناءً على ما شاركتِه، أنصحكِ بالتحدث مباشرة مع طبيب الأطفال أو القابلة "
            "بدلاً من الاعتماد على تجارب المجتمع وحدها. "
            "إذا كنتِ قلقة بشأن حالة طارئة، يرجى الاتصال بخدمات الطوارئ المحلية."
        ),
        "cta": "ابحثي عن طبيب أطفال قريب منكِ",
        "cta_url": "https://www.mumzworld.com/",
    },
}


async def handle_escalation(
    query: str,
    lang: str,
    urgency: str,
    topic: str,
    confidence: float,
    defer_reason: str = "low_confidence",
) -> Dict[str, Any]:
    """
    Phase 1: Returns static triage CTA.
    Phase 2 hook: add paediatric KB retrieval + partner directory lookup here.

    defer_reason: "low_confidence" | "seek_help_urgency" | "out_of_scope"
    """
    msg = ESCALATION_MESSAGES.get(lang, ESCALATION_MESSAGES["en"])
    secondary_lang = "ar" if lang == "en" else "en"
    secondary_msg  = ESCALATION_MESSAGES.get(secondary_lang, ESCALATION_MESSAGES["en"])

    return {
        "answer_primary":   f"{msg['title']}\n\n{msg['body']}",
        "answer_secondary": f"{secondary_msg['title']}\n\n{secondary_msg['body']}",
        "citations":        [],
        "urgency_flag":     urgency,
        "confidence_level": "deferred",
        "defer_message":    msg["body"],
        "hallucination_risk": False,
        "escalation_cta": {
            "label": msg["cta"],
            "url":   msg["cta_url"],
        },
    }


# ── LangGraph node wrapper ─────────────────────────────────────────────────────

async def escalation_node(state: dict) -> dict:
    """
    LangGraph node — async so it runs directly in FastAPI's event loop.
    LangGraph supports async nodes natively via ainvoke(); no thread hacks needed.
    """
    import logging as _logging
    lang       = state.get("lang_detected", "en")
    urgency    = state.get("urgency", "routine")
    topic      = state.get("topic", "health")
    confidence = state.get("confidence", 0.0)
    query      = state.get("query", "")

    try:
        result = await handle_escalation(query, lang, urgency, topic, confidence)
    except Exception as e:
        _logging.getLogger(__name__).error(f"escalation_node failed: {e}")
        msg = ESCALATION_MESSAGES.get(lang, ESCALATION_MESSAGES["en"])
        result = {
            "answer_primary":   f"{msg['title']}\n\n{msg['body']}",
            "answer_secondary": "",
            "citations":        [],
            "urgency_flag":     urgency,
            "confidence_level": "deferred",
            "defer_message":    msg["body"],
            "hallucination_risk": False,
        }

    return {**state, **result}