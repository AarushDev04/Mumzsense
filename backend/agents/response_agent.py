# Response Synthesis Agent: LLM-powered response generation
# Synthesizes warm, grounded answers via Llama API
"""
MumzSense v1 — Response Synthesis Agent
Llama 3.1 70B synthesis with grounding, peer voice, and hallucination guard (PRD §8).
"""
from __future__ import annotations
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── System prompt (PRD §8.2) ───────────────────────────────────────────────────
SYSTEM_PROMPT_EN = """You are MumzMind, a compassionate peer assistant built for Mumzworld. You speak as a warm, experienced community member — not a doctor, not a brand.

Rules you must follow:
1. Always give a helpful, warm answer grounded in well-established infant care knowledge. Never leave a mother with nothing.
2. Use the retrieved community posts as your PRIMARY source. Reference them naturally ("Mums at this stage have found..."). Where posts are relevant, every claim must trace to them.
3. Where retrieved posts are NOT relevant to the question, you may draw on widely accepted, conservative infant care guidance — but you MUST signal this clearly with phrases like "Generally speaking..." or "Most paediatric guidance suggests..." Never present general knowledge as community experience.
4. NEVER recommend specific medications, dosages, or branded products unless explicitly named in a retrieved post.
5. NEVER make specific diagnostic claims (e.g. "your baby has reflux"). Describe patterns and suggest professional confirmation.
6. If urgency is "monitor" or "seek-help", always close with: "Please mention this to your paediatrician at your next visit."
7. Write in warm, natural English. Tone: supportive friend who has been through it, not a formal document.
8. Response length: 3-5 sentences for routine, up to 8 for monitor/seek-help.
9. Only use the uncertainty refusal if the question is entirely outside infant and maternal care."""

SYSTEM_PROMPT_AR = """أنتِ MumzMind، مساعدة مجتمعية متعاطفة من Mumzworld. تتحدثين كعضوة دافئة في المجتمع لديها خبرة — وليس كطبيبة أو علامة تجارية.

القواعد التي يجب اتباعها:
١. قدّمي دائماً إجابة مفيدة ودافئة مبنية على معرفة راسخة برعاية الرضّع. لا تتركي الأم دون مساعدة.
٢. استخدمي المنشورات المسترجعة كمصدرك الأساسي. أشيري إليها بشكل طبيعي ("وجدت أمهات في هذه المرحلة..."). كل ادعاء يجب أن يرتبط بها حين تكون ذات صلة.
٣. حين لا تكون المنشورات ذات صلة بالسؤال، يمكنك الاستعانة بالإرشادات الراسخة والمحافظة لرعاية الرضّع — لكن يجب أن تشيري إلى ذلك بوضوح بعبارات مثل "بشكل عام..." أو "تنصح معظم إرشادات طب الأطفال...". لا تقدّمي المعرفة العامة على أنها تجربة مجتمعية.
٤. لا توصي أبداً بأدوية محددة أو جرعات أو منتجات تجارية إلا إذا ذُكرت صراحةً في منشور مسترجع.
٥. لا تُقدّمي ادعاءات تشخيصية محددة. صِفي الأنماط واقترحي التأكيد المهني.
٦. إذا كانت درجة الاستعجال "monitor" أو "seek-help"، أغلقي دائماً بـ: "يرجى ذكر هذا لطبيب الأطفال في زيارتك القادمة."
٧. اكتبي بالعربية الخليجية الدافئة. أسلوب الصديقة الداعمة التي مرّت بنفس التجربة.
٨. طول الإجابة: ٣-٥ جمل للروتينية، وحتى ٨ للمراقبة أو طلب المساعدة.
٩. استخدمي رفض عدم اليقين فقط إذا كان السؤال خارج نطاق رعاية الرضّع والأمومة كلياً."""


def _build_llm_client(provider: str):
    """Build LLM client based on provider env var."""
    from config import settings

    # Normalize provider and choose the correct API key
    prov = settings.effective_llm_provider()
    key = settings.effective_llm_key()

    if prov == "groq":
        from groq import Groq
        return Groq(api_key=key), settings.llama_model
    elif prov == "openrouter":
        from openai import OpenAI
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        ), "meta-llama/llama-3.1-70b-instruct:free"
    else:  # llama_api (default)
        from openai import OpenAI
        return OpenAI(
            base_url="https://api.llama.com/v1",
            api_key=key,
        ), settings.llama_model


def _build_user_prompt(
    query: str,
    posts: List[Dict],
    urgency: str,
    lang: str,
) -> str:
    """Construct the user turn for the LLM (PRD §8.3)."""
    post_blocks = []
    for i, p in enumerate(posts, 1):
        block = (
            f"Post {i} (Stage: {p.get('stage','?')}, Topic: {p.get('topic','?')}, "
            f"Trust: {p.get('trust_score', 0.75):.2f}):\n"
            f"Situation: {p.get('situation','')}\n"
            f"What helped: {p.get('advice','')}\n"
            f"Outcome: {p.get('outcome') or 'Not recorded'}"
        )
        post_blocks.append(block)

    posts_text = "\n\n".join(post_blocks)
    respond_in = "Arabic (Gulf/Khaleeji dialect)" if lang == "ar" else "English"

    return (
        f"Question: {query}\n\n"
        f"Community experiences retrieved (use where relevant, supplement with general infant care knowledge where needed):\n\n"
        f"{posts_text}\n\n"
        f"Urgency classification: {urgency}\n"
        f"Respond in: {respond_in}\n\n"
        f"Remember: Give a warm, helpful answer. If the posts above are not directly relevant, draw on your broad knowledge of infant and maternal care to help her."
    )


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    provider: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Single LLM call. Returns text content."""
    # Use normalized provider/key as defined in config
    from config import settings
    provider = settings.effective_llm_provider()
    client, model = _build_llm_client(provider)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content.strip()


def _hallucination_check(answer: str, posts: List[Dict]) -> bool:
    """
    Lightweight named-entity cross-check (PRD §8.5).
    Returns True if answer contains entities not in any retrieved post.
    """
    # Extract candidate entities: numbers, quoted terms, capitalized runs
    entities = set()
    # Numbers (weeks, temperatures, specific quantities)
    entities.update(re.findall(r"\b\d+\b", answer))
    # Quoted terms
    entities.update(re.findall(r'"([^"]+)"', answer))
    # Multi-word capitalized terms (product/drug names)
    entities.update(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", answer))

    # All text from retrieved posts
    corpus_text = " ".join(
        f"{p.get('situation','')} {p.get('advice','')} {p.get('outcome','')}"
        for p in posts
    )

    flagged = [e for e in entities if e and e not in corpus_text]
    risk = len(flagged) > 3  # tolerate minor number differences
    if risk:
        logger.warning(f"Hallucination risk — entities not in posts: {flagged[:5]}")
    return risk


UNCERTAINTY_RESPONSES = {
    "en": (
        "I don't have enough similar experiences in our community to answer this confidently. "
        "This might be something worth discussing directly with your healthcare provider, "
        "who can give advice specific to your baby's situation."
    ),
    "ar": (
        "ما عندي تجارب مشابهة كافية في مجتمعنا للإجابة على هذا بثقة. "
        "يمكن يكون هذا موضوع يستحق نقاشه مباشرة مع مقدم الرعاية الصحية لطفلك، "
        "اللي يقدر يعطيك نصيحة خاصة بوضع طفلك."
    ),
}

SEEK_HELP_SUFFIX = {
    "en": "\n\n⚠️ Given what you've described, please contact your paediatrician or healthcare provider today. If symptoms are severe, seek emergency care.",
    "ar": "\n\n⚠️ بناءً على ما وصفتِه، يرجى التواصل مع طبيب الأطفال أو مقدم الرعاية الصحية اليوم. إذا كانت الأعراض شديدة، اطلبي الرعاية الطارئة.",
}


async def synthesise(
    query: str,
    retrieved_posts: List[Dict],
    retrieval_confidence: str,
    topic: str,
    urgency: str,
    lang: str,
) -> Dict:
    """
    Main synthesis function. Returns response output schema (PRD §8.4).
    """
    from config import settings

    # ── Uncertainty path — only when zero posts retrieved ───────────────────
    if retrieval_confidence == "none" or not retrieved_posts:
        return {
            "answer_primary": UNCERTAINTY_RESPONSES.get(lang, UNCERTAINTY_RESPONSES["en"]),
            "answer_secondary": UNCERTAINTY_RESPONSES.get(
                "ar" if lang == "en" else "en",
                UNCERTAINTY_RESPONSES["en"]
            ),
            "citations": [],
            "urgency_flag": urgency,
            "confidence_level": "none",
            "defer_message": UNCERTAINTY_RESPONSES.get(lang, UNCERTAINTY_RESPONSES["en"]),
            "hallucination_risk": False,
        }

    # ── Normal synthesis path ───────────────────────────────────────────────
    sys_prompt  = SYSTEM_PROMPT_AR if lang == "ar" else SYSTEM_PROMPT_EN
    user_prompt = _build_user_prompt(query, retrieved_posts, urgency, lang)

    answer_primary = ""
    primary_fallback = False
    try:
        answer_primary = _call_llm(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            provider=settings.llm_provider,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
        )
    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        answer_primary = UNCERTAINTY_RESPONSES.get(lang, UNCERTAINTY_RESPONSES["en"])
        primary_fallback = True

    # Append seek-help suffix
    if urgency == "seek-help":
        answer_primary += SEEK_HELP_SUFFIX.get(lang, SEEK_HELP_SUFFIX["en"])

    # ── Secondary language synthesis ────────────────────────────────────────
    secondary_lang   = "ar" if lang == "en" else "en"
    secondary_sys    = SYSTEM_PROMPT_AR if secondary_lang == "ar" else SYSTEM_PROMPT_EN
    secondary_prompt = _build_user_prompt(query, retrieved_posts, urgency, secondary_lang)

    answer_secondary = ""
    if not primary_fallback:
        try:
            answer_secondary = _call_llm(
                system_prompt=secondary_sys,
                user_prompt=secondary_prompt,
                provider=settings.llm_provider,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
            )
            if urgency == "seek-help":
                answer_secondary += SEEK_HELP_SUFFIX.get(secondary_lang, "")
        except Exception as e:
            logger.warning(f"Secondary language synthesis failed (non-fatal): {e}")
            answer_secondary = UNCERTAINTY_RESPONSES.get(secondary_lang, "")

    # ── Hallucination guard ────────────────────────────────────────────────
    hallucination_risk = _hallucination_check(answer_primary, retrieved_posts)

    # ── Confidence level label ─────────────────────────────────────────────
    conf_map = {"high": "high", "medium": "medium", "low": "low", "none": "none"}
    confidence_level = conf_map.get(retrieval_confidence, "medium")

    citations = [p["post_id"] for p in retrieved_posts]

    return {
        "answer_primary":    answer_primary,
        "answer_secondary":  answer_secondary,
        "citations":         citations,
        "urgency_flag":      urgency,
        "confidence_level":  confidence_level,
        "defer_message":     None,
        "hallucination_risk": hallucination_risk,
    }


# ── LangGraph node wrapper ─────────────────────────────────────────────────────

async def response_node(state: dict) -> dict:
    """
    LangGraph node — async so it runs directly in FastAPI's event loop.
    LangGraph supports async nodes natively via ainvoke(); no thread hacks needed.
    """
    query                = state.get("query", "")
    retrieved_posts      = state.get("retrieved_posts", [])
    retrieval_confidence = state.get("retrieval_confidence", "none")
    topic                = state.get("topic", "health")
    urgency              = state.get("urgency", "routine")
    lang                 = state.get("lang_detected", "en")

    try:
        result = await synthesise(
            query, retrieved_posts, retrieval_confidence, topic, urgency, lang
        )
    except Exception as e:
        logger.error(f"response_node synthesise failed: {e}")
        fallback_msg = UNCERTAINTY_RESPONSES.get(lang, UNCERTAINTY_RESPONSES["en"])
        result = {
            "answer_primary":    fallback_msg,
            "answer_secondary":  UNCERTAINTY_RESPONSES.get("ar" if lang == "en" else "en", ""),
            "citations":         [],
            "urgency_flag":      urgency,
            "confidence_level":  "none",
            "defer_message":     None,
            "hallucination_risk": False,
        }

    return {
        **state,
        **result,
        # Expose retrieved_posts for citation building in main.py
        "retrieved_posts": retrieved_posts,
    }