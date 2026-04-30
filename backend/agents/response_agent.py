"""
Response Synthesis Agent — calls Groq to produce the bilingual answer.

Key fix (2026-04-30):
  Previous system prompt had a backdoor: "if posts do not clearly address
  the question, say so explicitly."  With hash embeddings, citations are
  ALWAYS partially off-topic, so Groq ALWAYS hit that branch and returned
  the defer message — even when latency was 2.6s (i.e. Groq was called).

  New behaviour:
  - Groq ALWAYS attempts an answer when confidence >= "low"
  - Posts are used as grounding context, not the only allowed source
  - General infant-care knowledge fills gaps, labelled "Generally speaking..."
  - Hard guardrails: no medications, no doses, no diagnoses, no branded products
  - defer_message is ONLY returned for non-maternal topics or confidence="none"
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Groq client initialisation
# ---------------------------------------------------------------------------

def _get_groq_client():
    try:
        from groq import Groq
        from app.config import get_settings  # type: ignore
        settings = get_settings()
        key = settings.groq_api_key
        model = settings.llama_model
        return Groq(api_key=key), model
    except Exception:
        pass
    try:
        from groq import Groq
        key = os.getenv("GROQ_API_KEY", "")
        model = os.getenv("LLAMA_MODEL", "llama-3.3-70b-versatile")
        return Groq(api_key=key), model
    except Exception as e:
        logger.error(f"Groq client init failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# System prompt — the definitive version
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are MumzSense, a warm, empathetic maternal health assistant for Mumzworld — the GCC region's leading baby and maternity platform.

You speak like a knowledgeable, caring friend who has been through the journey of motherhood. Your answers are grounded, practical, and emotionally supportive.

## ANSWER CONSTRUCTION RULES

**Rule 1 — Always attempt an answer.**
When you receive retrieved community posts, you MUST synthesise an answer. Do not refuse to answer just because the posts are not a perfect semantic match. Use the posts as context and grounding. Fill any gaps with well-established general infant-care knowledge.

**Rule 2 — Use posts as your primary source.**
When a retrieved post directly addresses the question, frame the answer as a community-sourced experience:
"Quite a few mums in our community have found that..."
"A mum who faced the same thing shared..."

**Rule 3 — Label general knowledge clearly.**
When you draw on general infant-care knowledge (not from the posts), prefix those statements with:
"Generally speaking, most paediatric guidance suggests..."
"In most cases, health visitors recommend..."
This is not a disclaimer — it is transparent, honest communication that helps the mother evaluate the source of the advice.

**Rule 4 — Urgency routing is non-negotiable.**
If the query involves ANY of: fever in a newborn under 3 months, non-blanching rash, difficulty breathing, seizure, unresponsiveness, severe vomiting, or signs of meningitis — you MUST:
1. Open with: "This needs immediate medical attention."
2. Advise calling emergency services or going to A&E now.
3. Keep the answer short — do not bury the urgency in a long response.
Set urgency_flag = "seek-help" for these.

**Rule 5 — Hard safety guardrails (ABSOLUTE — never violate).**
- Never name specific medications or dosages (e.g. "give 2.5ml of Calpol")
- Never name specific branded products as recommendations
- Never make a specific medical diagnosis ("your baby has reflux")
- Never speculate about rare conditions without recommending a professional
- For anything requiring a diagnosis, say: "A health visitor or paediatrician would be the right person to assess this in person."

**Rule 6 — The only valid reason to decline is scope.**
Only use the defer message if the question is completely outside maternal, infant, or child health (e.g. "what is the capital of France"). For ALL maternal/infant/child questions — sleep, feeding, development, health, behaviour, emotions — you MUST produce an answer, even if general.

## BILINGUAL OUTPUT FORMAT

Always respond with EXACTLY this JSON structure (no markdown, no preamble):
{
  "answer_primary": "<answer in the user's language — warm, 100-200 words>",
  "answer_secondary": "<same answer translated to the other language>",
  "urgency_flag": "routine" | "monitor" | "seek-help",
  "hallucination_risk": false
}

- answer_primary in English if query is English, Arabic if query is Arabic
- answer_secondary is the translation
- urgency_flag: routine = general advice, monitor = watch and check back, seek-help = go now
- hallucination_risk: set true ONLY if you had to speculate significantly beyond the posts and general knowledge

## WHAT "I DON'T HAVE ENOUGH EXPERIENCES" MEANS

This phrase is FORBIDDEN in your output except when the question has nothing to do with mothers or babies. If you find yourself about to write it for a maternal/infant question, stop and write a general-knowledge answer instead.
"""


DEFER_RESPONSE = {
    "answer_primary": (
        "I'm here to help with questions about your baby's health, feeding, sleep, and development. "
        "I wasn't able to find relevant guidance for this particular question — it may be outside "
        "my area of expertise. For this one, I'd suggest speaking with your healthcare provider "
        "or checking a trusted parenting resource."
    ),
    "answer_secondary": (
        "أنا هنا للمساعدة في الأسئلة المتعلقة بصحة طفلك والتغذية والنوم والنمو. "
        "لم أتمكن من العثور على إرشادات ذات صلة بهذا السؤال. "
        "للحصول على إجابة دقيقة، أنصحك بالتحدث مع مقدم الرعاية الصحية الخاص بك."
    ),
    "urgency_flag": "routine",
    "hallucination_risk": False,
    "defer_message": (
        "This question appears to be outside my area of maternal and infant health. "
        "Please consult a healthcare provider for personalised advice."
    ),
}


# ---------------------------------------------------------------------------
# Main synthesis function
# ---------------------------------------------------------------------------

def synthesise(
    query: str,
    posts: list[dict],
    confidence_level: str,
    lang: str = "en",
    topic: str | None = None,
    urgency: str | None = None,
) -> dict[str, Any]:
    """
    Call Groq to synthesise a bilingual answer.

    Returns a dict with: answer_primary, answer_secondary, urgency_flag,
                         hallucination_risk, defer_message (empty if answered).
    """

    # Hard defer: no posts at all
    if confidence_level == "none" or not posts:
        logger.warning("synthesise: confidence=none or no posts — returning defer")
        return {**DEFER_RESPONSE}

    groq_client, model = _get_groq_client()
    if groq_client is None:
        logger.error("synthesise: Groq client unavailable")
        return {
            **DEFER_RESPONSE,
            "defer_message": "Service temporarily unavailable. Please try again in a moment.",
        }

    # Build the user message with posts as context
    post_block = _format_posts(posts)
    user_message = f"""Mother's question: "{query}"
Baby stage hint: {topic or 'not specified'}
Urgency pre-classification: {urgency or 'not specified'}

Community experiences retrieved (use as grounding context):
{post_block}

Synthesise a warm, helpful answer following the system rules. Return ONLY valid JSON."""

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.4,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        import json
        result = json.loads(raw)

        # Validate required fields
        answer = result.get("answer_primary", "").strip()
        if not answer:
            logger.warning("synthesise: Groq returned empty answer_primary — using defer")
            return {**DEFER_RESPONSE}

        return {
            "answer_primary":   answer,
            "answer_secondary": result.get("answer_secondary", ""),
            "urgency_flag":     result.get("urgency_flag", "routine"),
            "hallucination_risk": bool(result.get("hallucination_risk", False)),
            "defer_message":    "",
        }

    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        return {
            **DEFER_RESPONSE,
            "defer_message": "Unable to generate a response at this time. Please try again.",
        }


# ---------------------------------------------------------------------------
# Helper: format retrieved posts into the user message
# ---------------------------------------------------------------------------

def _format_posts(posts: list[dict]) -> str:
    lines = []
    for i, p in enumerate(posts, 1):
        situation = p.get("situation", "")
        advice    = p.get("advice", "")
        outcome   = p.get("outcome", "")
        stage     = p.get("stage", "")
        topic     = p.get("topic", "")
        score     = p.get("similarity_score", 0)
        lang      = p.get("lang", "en")
        lines.append(
            f"[Post {i} | stage={stage} topic={topic} similarity={score:.3f} lang={lang}]\n"
            f"  Situation: {situation}\n"
            f"  Advice: {advice}\n"
            f"  Outcome: {outcome}"
        )
    return "\n\n".join(lines)