# MumzSense v1 — Product Requirements Document
### Phase 1: MumzMind RAG Core
**For: Mumzworld AI Engineering Intern Take-Home**
**Version:** 1.0 | **Date:** April 2026 | **Author:** Aarush

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview & Component Map](#2-system-overview--component-map)
3. [Step-by-Step Build Process](#3-step-by-step-build-process)
4. [Data Generation & Verification](#4-data-generation--verification)
5. [Data Preparation Pipeline](#5-data-preparation-pipeline)
6. [Classifier Agent — Decision Logic, Models & Ensemble](#6-classifier-agent--decision-logic-models--ensemble)
7. [RAG Pipeline — Vector Store & Hybrid Graph Extension](#7-rag-pipeline--vector-store--hybrid-graph-extension)
8. [Response Synthesis Agent](#8-response-synthesis-agent)
9. [LangGraph Orchestration](#9-langgraph-orchestration)
10. [Training & Testing Pipeline](#10-training--testing-pipeline)
11. [Evaluation Metrics & Test Parameters](#11-evaluation-metrics--test-parameters)
12. [Redis Caching Strategy](#12-redis-caching-strategy)
13. [FastAPI Endpoints](#13-fastapi-endpoints)
14. [Frontend Structure](#14-frontend-structure)
15. [Database Connectivity & Schema](#15-database-connectivity--schema)
16. [Tech Stack](#16-tech-stack)
17. [Repository Structure](#17-repository-structure)
18. [Docker & Containerised Delivery](#18-docker--containerised-delivery)
19. [Free Deployment Strategy](#19-free-deployment-strategy)
20. [How Components Tie Together — End-to-End Flow](#20-how-components-tie-together--end-to-end-flow)
21. [Next Steps to Final Product (Phase 2)](#21-next-steps-to-final-product-phase-2)

---

## 1. Executive Summary

MumzSense v1 is a RAG-powered maternal Q&A assistant for Mumzworld. A mother types a question in English or Arabic about her baby's health, feeding, or development. The system classifies the query, retrieves semantically similar experiences from a curated synthetic knowledge corpus, and synthesises a warm, grounded, peer-voiced answer via the Llama API — citing real post references and explicitly refusing when confidence is insufficient.

Phase 1 is the complete, self-contained, deployable core of Phase 2's multi-agent triage system (MumzSense full). Every architectural decision in Phase 1 is made with Phase 2 extension gates built in: the classifier slot is MADRL-extensible, the RAG layer is hybrid-graph-extensible, the LangGraph sequential graph is supervisor-extensible, and the escalation handler is a typed stub waiting for a real triage agent.

**Primary targets for Phase 1:**

- Functional bilingual (EN + AR) Q&A with grounded answers
- Honest uncertainty handling — system refuses when it shouldn't answer
- Classifier accuracy ≥ 82% on topic, ≥ 78% on urgency
- RAG retrieval precision@5 ≥ 0.70
- P95 response latency ≤ 4 seconds (with Redis cache hit: ≤ 300ms)
- Fully containerised, deployed on free infrastructure, accessible via public URL

---

## 2. System Overview & Component Map

```
User Query (EN/AR)
        │
        ▼
[Redis Cache Check] ──── HIT ──────────────────────────► Cached Response
        │ MISS
        ▼
[Classifier Agent]
  TF-IDF + BiLSTM ensemble
  Outputs: topic, urgency, lang, confidence, defer_flag
        │
        ├── defer_flag = TRUE ──► [Escalation Handler] ──► Static referral (v1)
        │                                                    ↕ Gate B (Phase 2)
        ▼
[RAG Agent]
  pgvector similarity search
  Metadata pre-filter: stage + topic
  bge-large embeddings (EN) / multilingual-mpnet (AR)
  Returns: top-5 posts + similarity scores
        │
        ├── max_score < 0.60 ──► [Uncertainty Response] ──► Honest refusal
        │
        ▼
[Response Synthesis Agent]
  Llama 3.1 70B via Llama API
  System prompt: peer-voiced, warm, cite posts, no hallucination
  Outputs: answer_en, answer_ar, citations[]
        │
        ▼
[Redis Cache Write]
        │
        ▼
Final Response → FastAPI → React Frontend
```

**Extension gates (built in Phase 1, activated in Phase 2):**

| Gate | Phase 1 State | Phase 2 Activation |
|------|--------------|-------------------|
| A — Classifier | BiLSTM policy head, logs to feedback_log | Replace with MADRL PPO policy |
| B — Escalation | Typed stub returning static CTA | Real triage agent + paediatric KB |
| C — LangGraph | Sequential graph, 3 nodes | Add supervisor, conditional edges |
| D — Hybrid RAG | pgvector only | Add Neo4j graph layer, dual retrieval |

---

## 3. Step-by-Step Build Process

The following is the sequential execution order. Each step has a clear deliverable and acceptance target before moving to the next.

### Step 1 — Environment & Infrastructure Setup (Target: 1.5 hours)
- Initialise monorepo with backend (FastAPI) and frontend (React + Vite) folders
- Provision Supabase project: enable pgvector extension, create schema
- Provision Upstash Redis: free tier, grab REST URL + token
- Set up `.env` with all secrets: Llama API key, Supabase URL + key, Redis URL
- Dockerise backend with a single `Dockerfile` and `docker-compose.yml`
- Confirm all services connect with a health-check endpoint
- **Acceptance:** `GET /health` returns 200 with db_status: ok, redis_status: ok

### Step 2 — Synthetic Corpus Generation (Target: 2 hours)
- Write `generate_corpus.py` using Llama API to produce 560 structured posts
- Run generation, validate every post against the JSON schema
- Reject and regenerate posts failing validation (target: < 5% rejection rate)
- Save validated corpus to `data/corpus_validated.jsonl`
- **Acceptance:** 560 posts, schema-valid, stage distribution within ±10% of target, AR posts readable by native speaker rubric

### Step 3 — EDA & Data Preparation (Target: 1 hour)
- Run EDA notebook: length distributions, topic balance, trust score distributions, AR/EN split
- Flag and fix any systematic generation artefacts
- Run preprocessing pipeline: clean, tokenise, feature-engineer
- **Acceptance:** EDA report saved, no class with < 30 training samples, corpus ready for embedding and classifier training

### Step 4 — Classifier Training (Target: 2 hours)
- Train TF-IDF baseline (topic + urgency, both heads)
- Train BiLSTM on same data
- Compute ensemble weights via Bayesian optimisation on validation set
- Save artefacts: vectoriser, BiLSTM weights, scaler, label encoders
- **Acceptance:** Ensemble topic accuracy ≥ 82%, urgency accuracy ≥ 78%, F1-macro ≥ 0.75 on held-out test set

### Step 5 — Embedding & pgvector Indexing (Target: 1 hour)
- Embed all 560 posts (EN: bge-large, AR: multilingual-mpnet)
- Push to Supabase pgvector with metadata columns
- Create HNSW index on embedding column
- **Acceptance:** All 560 posts indexed, similarity search returns results in < 200ms

### Step 6 — LangGraph Pipeline Assembly (Target: 2 hours)
- Implement `classifier_agent`, `rag_agent`, `response_agent`, `escalation_handler` as typed functions
- Wire sequential LangGraph graph with `AgentState` TypedDict
- Implement Redis cache check/write as graph entry/exit wrappers
- **Acceptance:** End-to-end pipeline returns a valid response for 5 manual test queries

### Step 7 — FastAPI App (Target: 1 hour)
- Implement all endpoints (see Section 13)
- Add input validation with Pydantic models
- Add error handling, logging, CORS configuration
- **Acceptance:** All endpoints return correct responses, Swagger docs accessible at `/docs`

### Step 8 — Frontend (Target: 2 hours)
- Build three screens: Landing, Chat, Uncertainty state
- Wire to FastAPI `/query` endpoint
- Display source cards with citations below each answer
- **Acceptance:** Full flow works in browser, AR text renders RTL correctly

### Step 9 — Evals (Target: 1.5 hours)
- Run `run_evals.py` against 20 test cases
- Compute all metrics, save to `evals/results.json`
- Write honest `EVALS.md` including failure modes
- **Acceptance:** All target metrics met or explained with honest tradeoff rationale

### Step 10 — Deployment (Target: 1 hour)
- Push Docker image to Railway / Render
- Deploy frontend on Vercel
- Confirm public URL works end-to-end
- **Acceptance:** Loom video recorded showing 5 live queries including one refusal

---

## 4. Data Generation & Verification

### 4.1 Corpus Specification

**Total posts:** 560
**Language split:** 440 English, 120 Arabic
**Stage distribution:**

| Stage | EN posts | AR posts |
|-------|----------|----------|
| Trimester | 50 | 15 |
| Newborn (0–4w) | 70 | 20 |
| 0–3 months | 90 | 25 |
| 3–6 months | 80 | 22 |
| 6–12 months | 80 | 22 |
| Toddler (12–24m) | 70 | 16 |

**Topic distribution (across all stages):**

| Topic | % of corpus |
|-------|-------------|
| feeding | 22% |
| sleep | 20% |
| health | 20% |
| development | 15% |
| gear | 10% |
| postpartum | 8% |
| mental_health | 5% |

**Urgency distribution:**

| Urgency | % of corpus |
|---------|-------------|
| routine | 55% |
| monitor | 30% |
| seek-help | 15% |

### 4.2 Post Schema

Every generated post must conform to this schema (validated with Pydantic before acceptance into corpus):

```
post_id:         UUID string
baby_age_weeks:  integer (0–104)
stage:           enum [trimester, newborn, 0-3m, 3-6m, 6-12m, toddler]
topic:           enum [feeding, sleep, health, development, gear, postpartum, mental_health]
urgency:         enum [routine, monitor, seek-help]
situation:       string, 20–200 chars, specific and concrete
advice:          string, 40–400 chars, first-person, peer-voiced
outcome:         string, 10–150 chars, or null
trust_score:     float 0.0–1.0
lang:            enum [en, ar]
verified:        bool (set to True after passing verification checks)
```

### 4.3 Generation Prompting Strategy

The Llama API is used with a structured generation prompt specifying:
- The exact JSON schema to output
- The persona: "You are a mother who went through this recently, writing to help others"
- The tone: warm, specific, non-clinical, peer-voiced
- Explicit instruction: "Do not invent medical advice. Express uncertainty naturally."
- Stage and topic are passed as generation parameters, forcing coverage of all cells in the distribution matrix
- AR posts generated with the instruction: "Write in natural conversational Gulf Arabic (Khaleeji), not Modern Standard Arabic"

Generation is batched in groups of 20 posts per API call to reduce latency. Each batch is validated immediately; failed posts are regenerated with a modified prompt that corrects the failure mode.

### 4.4 Data Verification for Trainability

After generation, every post passes through a verification pipeline before entering the corpus:

**Schema validation:** Pydantic model enforces all field types, enums, and length constraints. Rejection rate target < 5%.

**Language verification:** `langdetect` library confirms declared language matches detected language. AR posts are additionally checked for Arabic character presence (> 60% of tokens).

**Duplication check:** Cosine similarity between all post pairs using a lightweight embedding (sentence-transformers/all-MiniLM-L6-v2). Posts with similarity > 0.92 to any existing post are rejected to prevent memorisation artefacts.

**Class balance check:** After full generation, verify no topic-stage cell has fewer than 8 posts. If any cell is under-represented, trigger targeted generation to fill it.

**Trainability EDA checks:**
- Mean and variance of token length per class (flag degenerate classes with near-zero variance)
- Overlap of top-20 TF-IDF terms between classes (high overlap → weak separability → more generation needed)
- Trust score distribution (should not be bimodal — would indicate generation mode collapse)

**AR quality spot-check:** Manual review of 20 randomly sampled AR posts for fluency. If more than 3 of 20 read as literal translations, regenerate all AR posts with a stricter Arabic-native prompt.

---

## 5. Data Preparation Pipeline

### 5.1 Text Cleaning

Applied to both `situation` and `advice` fields before classifier training and embedding:

- Lowercase (EN only — preserve Arabic case)
- Remove URLs, email addresses, phone numbers
- Strip excess whitespace and newline characters
- Normalise Arabic text: remove diacritics (tashkeel), normalise alef variants (أ إ آ → ا), normalise taa marbuta
- Remove non-linguistic punctuation (keep sentence-ending punctuation for BiLSTM sequence modelling)
- For EN: expand common contractions (won't → will not) for better TF-IDF coverage

### 5.2 EDA — What to Measure

Before training, the following must be computed and saved to `data/eda_report.json`:

- Token length distribution per class (mean, std, min, max, P95) — used to set max_sequence_length for BiLSTM
- Vocabulary size after cleaning — used to set TF-IDF max_features
- Top-20 unigrams and bigrams per topic class — used to verify semantic separability
- Class distribution plots (topic, urgency, stage, lang) — verify no class < 30 samples
- Trust score histogram — verify roughly normal distribution (0.5–0.9 range)
- Inter-class cosine similarity matrix using mean TF-IDF vectors — flag any pair > 0.75

### 5.3 Feature Engineering

**For TF-IDF branch:**
- Input: concatenated `situation + " " + advice` string (cleaned)
- TF-IDF vectoriser: max_features=8000, ngram_range=(1,2), sublinear_tf=True, min_df=2
- Additional binary features appended to TF-IDF vector:
  - `has_medical_term`: 1 if any term from a 200-word medical keyword list appears
  - `has_urgency_signal`: 1 if terms like "fever", "emergency", "hospital", "not breathing" appear
  - `is_arabic`: 1/0 based on langdetect
  - `text_length_bin`: 0/1/2 (short/medium/long) based on token count

**For BiLSTM branch:**
- Input: tokenised sequence of `situation + advice` (post-cleaning)
- Vocabulary: built from training corpus, size capped at 12,000 tokens
- Embedding layer: 128-dim, trained from scratch (small corpus, pretrained embeddings would overfit)
- Sequence padding: right-pad to P95 length from EDA (typically ~90 tokens)
- Unknown token: `<UNK>` for words outside vocabulary

**Target labels:**
- Topic: 7-class multiclass (one-hot encoded)
- Urgency: 3-class multiclass (one-hot encoded)
- Both heads trained simultaneously in a multi-task BiLSTM

### 5.4 Train/Validation/Test Split

- Training: 70% (392 posts)
- Validation: 15% (84 posts) — used for hyperparameter tuning and ensemble weight optimisation
- Test: 15% (84 posts) — held out until final evaluation, never used during training
- Stratified split on topic × urgency joint label to preserve class distribution
- Split performed before any augmentation so test set is strictly unseen

---

## 6. Classifier Agent — Decision Logic, Models & Ensemble

### 6.1 Design Rationale

The classifier must solve two simultaneous problems: topic classification (7 classes) and urgency classification (3 classes). The ensemble of TF-IDF + BiLSTM is chosen deliberately over a single transformer because:

- TF-IDF is fast at inference (< 5ms), interpretable, and strong on keyword-distinctive topics (feeding vs gear vs health have very different vocabulary)
- BiLSTM captures sequential context that TF-IDF misses: "sleep was fine until the regression started" vs "won't sleep at all" have similar keywords but different urgency signals
- The combined model is trainable in under 20 minutes on CPU, which matters for the 1.5-day constraint
- The BiLSTM hidden state is a fixed-dimensional vector that can be replaced by a MADRL policy head in Phase 2 without changing the downstream interface (Gate A)
- A transformer fine-tune (e.g. AraBERT + mBERT) would be more accurate but requires GPU, longer training, and is overkill for a 560-post corpus

### 6.2 TF-IDF Baseline Model

**Architecture:** Logistic Regression with L2 regularisation on top of TF-IDF + engineered features

**Topic head:**
- Multi-class Logistic Regression, solver=lbfgs, max_iter=1000, C=1.0
- Input: 8000-dim TF-IDF vector + 4 engineered binary features = 8004-dim
- Output: 7-class probability distribution (softmax)

**Urgency head:**
- Same architecture, 3-class output
- Trained independently (not multi-task) for simplicity at this stage

**Why Logistic Regression over SVM or Random Forest:**
- Outputs calibrated probabilities natively (needed for ensemble weighting)
- Faster inference than Random Forest on high-dimensional TF-IDF vectors
- SVM requires Platt scaling for probability outputs — extra complexity not needed

### 6.3 BiLSTM Neural Model

**Architecture (multi-task):**

```
Input sequence (padded tokens)
        │
[Embedding Layer: vocab_size × 128]
        │
[Bidirectional LSTM: 128 units per direction → 256-dim output]
        │
[Dropout: 0.3]
        │
[Dense: 128 units, ReLU]
        │
    ┌───┴───┐
    │       │
[Topic    [Urgency
 head:     head:
 Dense 7   Dense 3
 Softmax]  Softmax]
```

**Training configuration:**
- Loss: categorical crossentropy on both heads, equal weighting (0.5 + 0.5)
- Optimiser: Adam, lr=0.001, decay to 0.0001 after 5 epochs
- Batch size: 32
- Epochs: 25 with early stopping (patience=5, monitor=val_loss)
- Regularisation: Dropout 0.3 on LSTM output, L2=0.001 on Dense layers
- Framework: Keras (TensorFlow backend) — chosen for simplicity and portability

**Why BiLSTM over plain LSTM:**
- Bidirectional processing captures context from both directions in the post text
- In maternal health posts, the situation is often described before the symptom: "We were having a normal night when suddenly she started [urgent symptom]" — the urgency signal appears late in the sequence, which a forward-only LSTM underweights

**Why 128-dim embedding trained from scratch:**
- Pre-trained embeddings (GloVe, FastText) are trained on general corpora; maternal health vocabulary ("cluster feeding", "sleep regression", "fontanelle") is underrepresented
- 560 posts is small but sufficient to learn domain-specific representations at 128 dims
- Arabic pre-trained embeddings introduce licensing and compatibility complexity not justified at this scale

### 6.4 Ensemble Strategy & Weighting

**Ensemble method:** Weighted probability averaging (soft voting)

The final probability vector for each head is:

```
P_ensemble(class) = w_tfidf × P_tfidf(class) + w_bilstm × P_bilstm(class)
```

**Weight optimisation:** Bayesian optimisation (scikit-optimize BayesSearchCV) on the validation set, searching the space w_tfidf ∈ [0.2, 0.8], w_bilstm = 1 − w_tfidf. This is a 1-dimensional search; Bayesian optimisation finds the optimum in 20–30 evaluations vs 100 grid search steps.

**Expected weight range from prior experiments with similar hybrid classifiers:**
- Topic classification: TF-IDF contributes more (~0.55–0.65) because topic words are highly distinctive
- Urgency classification: BiLSTM contributes more (~0.55–0.65) because urgency is expressed sequentially

**Confidence scoring:**
- Raw: max probability from ensemble output
- Calibrated: Platt scaling applied to ensemble outputs using validation set (sklearn CalibratedClassifierCV)
- Confidence threshold for defer_flag: 0.60 (if max_prob < 0.60, defer = True regardless of predicted class)
- Additional defer trigger: urgency prediction == "seek-help" always sets defer = True

**Classifier output schema:**

```
{
  "topic": "health",
  "topic_confidence": 0.87,
  "urgency": "monitor",
  "urgency_confidence": 0.79,
  "lang_detected": "en",
  "defer_flag": false,
  "raw_probs_topic": { "feeding": 0.04, "sleep": 0.03, "health": 0.87, ... },
  "raw_probs_urgency": { "routine": 0.14, "monitor": 0.79, "seek-help": 0.07 }
}
```

### 6.5 Language Detection

- Primary: `langdetect` library (runs in < 1ms)
- Fallback: Arabic Unicode character ratio — if > 30% of characters are Arabic Unicode points, classify as AR regardless of langdetect output
- Language detection is independent of classification and runs before the classifier to allow language-specific preprocessing

### 6.6 MADRL Extension Gate A

The BiLSTM's 256-dim hidden state (output of the LSTM layer before the Dense layer) is logged alongside the classification decision and the eventual user feedback signal to the `feedback_log` table. In Phase 2, this hidden state becomes the state representation for the MADRL policy, and the Dense head is replaced by a PPO policy head. The interface to the downstream graph node remains identical: it still receives the same output schema. This is the gate — the classifier is already behaving as a policy; it just isn't learning from feedback yet.

---

## 7. RAG Pipeline — Vector Store & Hybrid Graph Extension

### 7.1 Embedding Models

**English posts and queries:**
- Model: `BAAI/bge-large-en-v1.5` (HuggingFace, free)
- Dimension: 1024
- Chosen for: state-of-the-art performance on MTEB retrieval benchmarks for English, especially for short query → longer document retrieval (asymmetric retrieval — exactly the maternal Q&A use case)
- Query prefix: "Represent this question for searching relevant maternal experiences: " (bge-large requires an instruction prefix for queries)

**Arabic posts and queries:**
- Model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (HuggingFace, free)
- Dimension: 768
- Chosen for: strong Arabic performance, multilingual alignment (Arabic queries can retrieve relevant EN posts if needed), runs on CPU in < 200ms per batch of 50

**Dual-column storage:** Both EN and AR embeddings stored in separate pgvector columns on the same posts table, allowing the retrieval function to select the right column based on `lang_detected` from the classifier.

### 7.2 pgvector Configuration

**Supabase pgvector setup:**
- Extension: `pgvector` (enabled by default on Supabase)
- Index type: HNSW (Hierarchical Navigable Small World) — chosen over IVFFlat because HNSW gives better recall at small corpus sizes (< 100K vectors) and does not require a training phase
- HNSW parameters: `m=16, ef_construction=64` (Supabase defaults — suitable for 560 vectors)
- Distance metric: cosine distance (L2-normalised vectors — cosine similarity = dot product after normalisation)

**Retrieval procedure:**

1. Pre-filter: SQL WHERE clause on `stage` (if determinable from classifier output) and `topic` (from classifier). This reduces the candidate pool before vector search, improving both precision and latency.
2. Vector search: `ORDER BY embedding <=> query_embedding LIMIT 10` (cosine distance operator in pgvector)
3. Post-filter: keep only results with similarity score > 0.45 (i.e. cosine distance < 0.55)
4. Return top-5 after post-filtering

**Similarity threshold logic:**
- top_score > 0.75: high confidence retrieval, proceed normally
- top_score 0.60–0.75: medium confidence, add a hedging phrase to the synthesised response
- top_score 0.45–0.60: low confidence, retrieved posts shown as "loosely related" in UI
- top_score < 0.45 OR fewer than 2 results above threshold: trigger uncertainty response, do not synthesise

### 7.3 Hybrid RAG — Graph Extension Gate D

This is the architectural gate for Phase 2's knowledge graph layer. The vector retrieval returns semantically similar posts. A knowledge graph layer would additionally traverse explicit relationships between posts, topics, and outcomes — enabling answers like "mothers who had this feeding issue at 8 weeks often encountered X sleep issue at 12 weeks."

**Phase 1:** pgvector only. All retrieval goes through the vector path.

**Phase 2 extension — GraphRAG layer:**
- Graph database: Neo4j AuraDB free tier (or Memgraph for self-hosted)
- Nodes: Post, Stage, Topic, Outcome, Author (anonymised)
- Relationships: `POST -[BELONGS_TO]-> STAGE`, `POST -[TAGGED]-> TOPIC`, `POST -[LED_TO]-> OUTCOME`, `POST -[SIMILAR_TO {score}]-> POST` (pre-computed for similarity > 0.8)
- Retrieval: Cypher query traverses from matched topic node through SIMILAR_TO and LED_TO edges to discover outcome chains not reachable by pure vector similarity
- Dual retrieval: vector retrieval results + graph traversal results are merged with a weighted score (vector_score × 0.6 + graph_relevance × 0.4)
- The `rag_agent.py` function signature in Phase 1 already accepts an optional `use_graph=False` parameter, which is the extension point

**Why this matters:** Vector similarity finds posts that sound like the query. Graph traversal finds posts that are causally or developmentally connected to the query context. A mother asking about 3-month sleep regression might benefit from posts about 4-month regression that are connected via developmental outcome edges — a connection a pure vector search would miss.

### 7.4 Retrieval Output Schema

```
{
  "retrieved_posts": [
    {
      "post_id": "uuid",
      "situation": "...",
      "advice": "...",
      "outcome": "...",
      "trust_score": 0.84,
      "similarity_score": 0.81,
      "stage": "0-3m",
      "topic": "feeding",
      "lang": "en"
    },
    ...
  ],
  "retrieval_confidence": "high",
  "max_similarity": 0.81,
  "query_lang": "en"
}
```

---

## 8. Response Synthesis Agent

### 8.1 Llama API Configuration

- Model: `meta-llama/Llama-3.1-70B-Instruct` via Llama API (free tier)
- Max tokens: 600 (sufficient for a warm, detailed response without padding)
- Temperature: 0.4 (low enough for factual grounding, high enough for natural peer voice)
- Top-p: 0.85

### 8.2 System Prompt Design

The system prompt is the most important engineering artefact in the synthesis agent. It must enforce three constraints simultaneously: grounding (only use retrieved posts), voice (warm, peer, not clinical), and honesty (express uncertainty, never invent).

**System prompt structure:**

```
You are MumzMind, a compassionate peer assistant built for Mumzworld. You speak as a knowledgeable community member, not a doctor or brand. Your answers must be grounded entirely in the posts provided to you. Never add information not present in the posts.

Rules you must follow without exception:
1. Every claim in your response must trace to one of the provided posts. Reference posts by number (e.g. "Three mothers at this stage found that...").
2. If the posts do not clearly address the question, say so explicitly: "I don't have enough similar experiences to answer this confidently."
3. Never use clinical language unless it appears directly in a post.
4. Never recommend specific products unless explicitly mentioned in a post.
5. If urgency is "monitor" or "seek-help", always close with: "Please mention this to your paediatrician at your next visit" (or equivalent in Arabic).
6. Write in {lang}: {language_instruction}.
7. Response length: 3–5 sentences for routine queries, up to 8 sentences for monitor/seek-help queries.
```

**Language instruction for Arabic:**
"Write in warm, conversational Gulf Arabic (Khaleeji dialect where natural). Do not use Modern Standard Arabic. Keep the tone of a supportive friend, not a formal document."

### 8.3 Prompt Construction

The user turn sent to the API is constructed as:

```
Question: {user_query}

Relevant experiences from our community (use these as your only source):

Post 1 (Stage: {stage}, Topic: {topic}, Trust: {trust_score}):
Situation: {situation}
What helped: {advice}
Outcome: {outcome}

Post 2: [...]
...Post 5: [...]

Urgency classification: {urgency}
Respond in: {lang}
```

### 8.4 Output Schema

```
{
  "answer_primary": "...",        # Answer in query language
  "answer_secondary": "...",      # Answer in other language (always provided)
  "citations": ["post_1_id", "post_3_id"],
  "urgency_flag": "monitor",
  "confidence_level": "medium",
  "defer_message": null           # Populated when no good retrieval
}
```

### 8.5 Hallucination Guard

After synthesis, a lightweight post-processing check verifies that named entities in the response (product names, medical terms, specific numbers like "at 8 weeks") appear in at least one of the retrieved posts. If a named entity in the response does not appear in any retrieved post, the response is flagged with `hallucination_risk: true` and the UI shows a soft warning. This check uses simple string matching — not a second LLM call — to keep latency low.

---

## 9. LangGraph Orchestration

### 9.1 Graph Structure (Phase 1 — Sequential)

LangGraph is used as the orchestration layer even in Phase 1's sequential execution. This is deliberate: the graph structure is the extension gate for Phase 2's supervisor architecture. The Phase 1 graph is a straight line; Phase 2 adds routing edges without changing any node implementation.

**AgentState TypedDict:**

```
class AgentState(TypedDict):
    query: str
    lang_detected: str
    topic: str
    urgency: str
    confidence: float
    defer_flag: bool
    retrieved_posts: list
    retrieval_confidence: str
    answer_primary: str
    answer_secondary: str
    citations: list
    hallucination_risk: bool
    cached: bool
    error: Optional[str]
```

**Phase 1 graph nodes (sequential):**

```
START
  → cache_check_node
  → classifier_node        (Gate A slot)
  → defer_router           (conditional: if defer_flag → escalation_node, else → rag_node)
  → rag_node               (Gate C slot)
  → threshold_router       (conditional: if retrieval_confidence == "none" → uncertainty_node, else → synthesis_node)
  → synthesis_node
  → cache_write_node
END
```

**Phase 2 additions (without changing existing nodes):**
- Add `supervisor_node` between `classifier_node` and the existing agent nodes
- Add `triage_node` (Gate B) as a new branch from `defer_router`
- Add conditional edges based on topic and urgency to route to specialist agents
- The existing `rag_node` and `synthesis_node` become leaf nodes in the supervisor graph

### 9.2 Error Handling Within the Graph

Each node wraps its execution in a try/except. On exception, the node sets `state["error"]` and returns. A global error router at the end of the graph checks for `state["error"]` and returns a graceful fallback response to the user rather than a 500 error. Errors are logged with the full state for debugging.

---

## 10. Training & Testing Pipeline

### 10.1 Training Pipeline Sequence

```
1. Load corpus_validated.jsonl
2. Run cleaning pipeline (Section 5.1)
3. Run EDA, save eda_report.json
4. Split: 70/15/15 stratified
5. TF-IDF branch:
   a. Fit TF-IDF vectoriser on train set
   b. Compute engineered features
   c. Train Logistic Regression (topic head) on train set
   d. Train Logistic Regression (urgency head) on train set
   e. Evaluate on validation set, save val_metrics_tfidf.json
6. BiLSTM branch:
   a. Build vocabulary from train set
   b. Tokenise and pad sequences
   c. Train multi-task BiLSTM with early stopping
   d. Evaluate on validation set, save val_metrics_bilstm.json
7. Ensemble:
   a. Generate soft probability outputs from both models on validation set
   b. Run Bayesian weight optimisation (20–30 evaluations)
   c. Apply calibration (Platt scaling) on validation set
   d. Save optimal weights and calibration parameters
8. Final evaluation on held-out test set (one-time only)
9. Save all artefacts to models/ directory
```

### 10.2 Embedding Pipeline Sequence

```
1. Load corpus_validated.jsonl
2. Run cleaning pipeline
3. For each post:
   a. Detect language
   b. If EN: embed with bge-large, store in en_embedding column
   c. If AR: embed with multilingual-mpnet, store in ar_embedding column
   d. If EN post but also want AR retrieval: embed with multilingual-mpnet, store in ar_embedding
4. Insert all posts with embeddings into Supabase posts table
5. Create HNSW index on en_embedding and ar_embedding columns
6. Run 10 test queries, verify top-5 results are semantically relevant
```

### 10.3 Test Case Design

**20 test cases minimum for the evaluation harness:**

| Category | Count | Description |
|----------|-------|-------------|
| Easy EN | 5 | Clear topic, routine urgency, sufficient corpus coverage |
| Easy AR | 3 | Same as above in Arabic |
| Adversarial — vague | 3 | Very short or ambiguous queries that should trigger deferral |
| Adversarial — urgency | 3 | Queries with medical red flags (fever, not breathing, seizure) that must trigger seek-help |
| Adversarial — out-of-scope | 2 | Non-maternal queries (politics, recipes) — must defer or refuse |
| Edge — mixed lang | 2 | Queries mixing EN and AR in same message |
| Edge — no corpus match | 2 | Topics genuinely not in corpus — must return uncertainty response |

---

## 11. Evaluation Metrics & Test Parameters

### 11.1 Classifier Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Topic accuracy (macro) | ≥ 82% | sklearn accuracy_score on test set |
| Topic F1-macro | ≥ 0.75 | sklearn f1_score(average='macro') |
| Urgency accuracy (macro) | ≥ 78% | sklearn accuracy_score on test set |
| Urgency F1-macro | ≥ 0.72 | sklearn f1_score(average='macro') |
| Seek-help recall | ≥ 0.90 | Critical: must not miss urgent cases — sklearn recall for "seek-help" class |
| Confidence calibration | ECE ≤ 0.10 | Expected Calibration Error after Platt scaling |
| Deferral on vague queries | 100% | All 3 vague test cases must trigger defer_flag = True |

**Why seek-help recall is the most important single metric:** A false negative on urgency (classifying a "seek-help" case as "routine") is a safety failure. The threshold for this class is deliberately asymmetric — we accept lower precision (more false positives that get deferred) in exchange for near-perfect recall on genuine urgent cases.

### 11.2 RAG Retrieval Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Precision@5 | ≥ 0.70 | Human-judged relevance of top-5 retrieved posts on 10 test queries |
| Recall@10 | ≥ 0.65 | Coverage of relevant posts in top-10 |
| MRR (Mean Reciprocal Rank) | ≥ 0.75 | Mean of 1/rank of first relevant post |
| Retrieval latency P95 | ≤ 200ms | Measured on Supabase with HNSW index |
| Correct deferral on no-match queries | 100% | All 2 no-match test cases must return uncertainty response |

**Human relevance judgement rubric (for Precision@5):**
- Score 2: Post directly addresses the query situation and provides actionable advice
- Score 1: Post is loosely related (same topic, different age/situation)
- Score 0: Post is unrelated
- Precision@5 = (number of posts scored ≥ 1) / 5

### 11.3 Response Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Grounding rate | ≥ 95% | % of factual claims traceable to retrieved posts (manual spot-check) |
| Hallucination flag rate | ≤ 5% | % of responses flagged by post-processing hallucination check |
| AR naturalness | ≥ 4/5 | Human rubric score on 5-point scale for AR responses |
| Refusal on out-of-scope | 100% | All out-of-scope test cases return uncertainty or deferral |
| Cite rate | ≥ 90% | % of responses that include at least one post citation |

### 11.4 System Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-end P95 latency (cold) | ≤ 4 seconds | Artillery load test, 10 concurrent users |
| End-to-end latency (cache hit) | ≤ 300ms | Artillery test with repeated queries |
| Cache hit rate (repeated queries) | 100% | Send same query twice, verify second is served from cache |
| API uptime | ≥ 99% during demo | Manual monitoring |

---

## 12. Redis Caching Strategy

### 12.1 Cache Provider

**Upstash Redis** (free tier): 10,000 commands/day, 256MB storage — sufficient for demo and evaluation. Accessed via REST API (no TCP connection needed, works on serverless deployments).

### 12.2 Cache Key Design

Cache keys are constructed to balance specificity and hit rate:

```
mumzsense:query:{sha256(query_normalised + stage_hint)}
```

**Query normalisation before hashing:**
- Lowercase (EN queries)
- Strip punctuation except Arabic punctuation
- Strip leading/trailing whitespace
- Collapse repeated whitespace
- Do NOT stem or lemmatise (changes meaning in maternal health context)

**Stage hint** is included in the key because the same query ("won't sleep") has different answers at newborn vs 6-month stage. If the user has selected a stage in the UI, it is included in the hash. If not, the query is cached without stage (lower precision, higher hit rate).

### 12.3 Cache TTL Strategy

| Query type | TTL | Rationale |
|-----------|-----|-----------|
| Routine queries | 24 hours | Stable answers, high repeat rate |
| Monitor queries | 6 hours | May need freshness if corpus updates |
| Seek-help responses | NOT CACHED | Safety: always re-evaluate, never serve stale urgent guidance |
| Uncertainty responses | 1 hour | Short TTL — corpus may grow to answer these |
| Deferred responses | NOT CACHED | Always re-run classifier in case urgency changed |

### 12.4 Cache Invalidation

- On any corpus update (new posts added, posts removed): flush `mumzsense:query:*` keys
- On classifier model update: flush all cached responses (model may classify differently)
- Manual flush endpoint: `POST /admin/cache/flush` (auth-protected)

### 12.5 Cache Data Structure

Stored as JSON string:

```json
{
  "answer_primary": "...",
  "answer_secondary": "...",
  "citations": ["post_id_1"],
  "urgency_flag": "routine",
  "confidence_level": "high",
  "cached_at": "2026-04-27T14:23:00Z",
  "cache_version": "v1.2"
}
```

---

## 13. FastAPI Endpoints

### 13.1 Core Endpoints

**`POST /query`**
Primary query endpoint. Runs the full LangGraph pipeline.

Request body:
```json
{
  "query": "my 8 week old has been cluster feeding for 3 hours, is this normal?",
  "stage_hint": "0-3m",       // optional, from UI stage selector
  "lang_preference": "en"     // optional, overrides auto-detection
}
```

Response body:
```json
{
  "answer_primary": "...",
  "answer_secondary": "...",
  "citations": [...],
  "urgency_flag": "routine",
  "confidence_level": "high",
  "defer_message": null,
  "hallucination_risk": false,
  "cached": false,
  "latency_ms": 2340
}
```

Error responses:
- 422: Invalid request body (Pydantic validation)
- 503: Llama API unavailable (with graceful fallback message)
- 200 with `defer_message` populated: system chose not to answer

**`GET /health`**
Returns system health status. Used by deployment platform for uptime monitoring.

Response:
```json
{
  "status": "ok",
  "db_status": "ok",
  "redis_status": "ok",
  "model_loaded": true,
  "corpus_size": 560,
  "version": "1.0.0"
}
```

**`GET /corpus/stats`**
Returns corpus metadata for display in the UI's "About" section.

**`POST /feedback`**
Logs user feedback for Gate A (MADRL preparation).

Request body:
```json
{
  "query_hash": "sha256...",
  "rating": 4,               // 1-5
  "was_helpful": true,
  "urgency_felt": "routine"  // user's self-assessment
}
```

**`POST /admin/cache/flush`**
Auth-protected (Bearer token). Flushes Redis cache.

**`GET /evals/latest`**
Returns the latest evaluation run results from `evals/results.json`.

### 13.2 Pydantic Models

All request/response bodies are defined as Pydantic models in `backend/schemas.py`. This provides automatic validation, OpenAPI documentation, and type safety across the entire API surface.

### 13.3 CORS Configuration

Allowed origins: the Vercel frontend URL + `localhost:5173` (Vite dev server). All other origins blocked.

### 13.4 Rate Limiting

Per-IP rate limiting using `slowapi` (FastAPI-compatible): 20 requests/minute per IP. This protects the free Llama API tier from abuse during the demo period.

---

## 14. Frontend Structure

### 14.1 Tech Stack

- React 18 + Vite (fast HMR, minimal config)
- Tailwind CSS (utility-first, consistent with Mumzworld's design language)
- `react-query` (TanStack Query) for API calls, loading states, and cache
- `i18next` for EN/AR internationalisation and RTL layout switching
- No component library — custom components to keep bundle small

### 14.2 Screen Inventory

**Screen 1 — Landing**
- Mumzworld branding (logo, colour palette: #FF6B6B coral, white, warm grey)
- Headline: "Ask anything about your baby's journey"
- Subheadline: "Answers from mothers who've been exactly where you are"
- Stage selector: pill buttons (Trimester, Newborn, 0–3m, 3–6m, 6–12m, Toddler)
- Language toggle: EN / AR (switches UI language and input placeholder)
- Single text input with "Ask" button
- Trust signal: "Powered by 560+ real mother experiences"

**Screen 2 — Chat Thread**
- Message bubbles: user queries (right-aligned), MumzMind answers (left-aligned)
- Each answer followed by expandable SourceCards (collapsed by default, count shown)
- SourceCard shows: stage, topic, situation summary, advice excerpt, trust score bar
- Urgency badge on answer if urgency == "monitor" or "seek-help" (amber/red pill)
- Loading skeleton during API call (P95 = 4s — users need visual feedback)
- "Doctor referral" CTA card (soft teal background) appears below seek-help answers
- Arabic answers render RTL with appropriate font (Cairo or Tajawal from Google Fonts)

**Screen 3 — Uncertainty State**
- Triggered when `defer_message` is populated
- Soft grey background, honest message: "I don't have enough similar experiences to answer this confidently"
- Suggestion: "You might find better answers in our community forum" (Phase 2 hook)
- Option to rephrase or try a different stage

**Screen 4 — Error State**
- Gentle, branded error message if API call fails
- Retry button, does not show raw error to user

### 14.3 State Management

- Global state: React Context for language preference and selected stage
- Query state: TanStack Query handles loading, error, and stale states
- Chat history: useState array of `{role, content, citations, metadata}` — in-memory only (no persistence in Phase 1)

### 14.4 RTL Support

When language is Arabic:
- `dir="rtl"` on root div
- Tailwind's RTL plugin handles margin/padding mirroring
- Text input placeholder switches to Arabic
- Arabic font (Cairo) loaded via Google Fonts (subsetted for performance)
- Bubble alignment reverses (user: left, MumzMind: right in RTL)

### 14.5 Performance

- Code splitting by route (React.lazy)
- Source card content lazy-loaded on expand
- API responses cached by TanStack Query for 5 minutes client-side (reduces redundant API calls during same session)
- Vite build output: target < 150KB gzipped

---

## 15. Database Connectivity & Schema

### 15.1 Supabase Connection

- Connection via Supabase Python client (`supabase-py`) in FastAPI backend
- Connection pooling: PgBouncer (provided by Supabase, no configuration needed)
- All queries use parameterised statements (no string interpolation — SQL injection prevention)
- pgvector queries executed via raw SQL through the Supabase `rpc()` call or direct `asyncpg` connection

### 15.2 Schema

**Table: `posts`**

```sql
CREATE TABLE posts (
  post_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  baby_age_weeks  INTEGER NOT NULL,
  stage           TEXT NOT NULL CHECK (stage IN ('trimester','newborn','0-3m','3-6m','6-12m','toddler')),
  topic           TEXT NOT NULL CHECK (topic IN ('feeding','sleep','health','development','gear','postpartum','mental_health')),
  urgency         TEXT NOT NULL CHECK (urgency IN ('routine','monitor','seek-help')),
  situation       TEXT NOT NULL,
  advice          TEXT NOT NULL,
  outcome         TEXT,
  trust_score     FLOAT NOT NULL DEFAULT 0.75,
  lang            TEXT NOT NULL CHECK (lang IN ('en','ar')),
  en_embedding    vector(1024),   -- bge-large dimension
  ar_embedding    vector(768),    -- multilingual-mpnet dimension
  verified        BOOLEAN DEFAULT FALSE,
  created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX posts_en_embedding_hnsw ON posts
  USING hnsw (en_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX posts_ar_embedding_hnsw ON posts
  USING hnsw (ar_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX posts_topic_stage ON posts (topic, stage);
```

**Table: `feedback_log`** (Gate A — MADRL preparation)

```sql
CREATE TABLE feedback_log (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  query_hash      TEXT NOT NULL,
  query_text      TEXT NOT NULL,
  lang_detected   TEXT,
  topic_predicted TEXT,
  urgency_predicted TEXT,
  classifier_confidence FLOAT,
  bilstm_hidden_state JSONB,   -- 256-dim vector, logged for Phase 2 MADRL
  retrieved_post_ids TEXT[],
  max_similarity  FLOAT,
  answer_given    TEXT,
  user_rating     INTEGER,
  was_helpful     BOOLEAN,
  created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

**Table: `query_cache_log`** (analytics, not the Redis cache itself)

```sql
CREATE TABLE query_cache_log (
  query_hash      TEXT PRIMARY KEY,
  query_normalised TEXT,
  cache_hits      INTEGER DEFAULT 0,
  last_hit        TIMESTAMPTZ,
  created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 16. Tech Stack

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| LLM | Llama 3.1 70B via Llama API | Latest | Free tier, strong instruction following, multilingual |
| Orchestration | LangGraph | 0.1.x | Agent graph with clean extension to Phase 2 supervisor |
| Backend framework | FastAPI | 0.111+ | Async, typed, OpenAPI auto-docs |
| Classifier (TF-IDF) | scikit-learn | 1.4+ | TF-IDF vectoriser + LogReg |
| Classifier (BiLSTM) | Keras / TensorFlow | 2.16+ | Multi-task LSTM, exportable |
| Embedding (EN) | sentence-transformers (bge-large) | 2.7+ | MTEB SOTA for asymmetric retrieval |
| Embedding (AR) | sentence-transformers (multilingual-mpnet) | 2.7+ | Strong Arabic support |
| Vector DB | pgvector on Supabase | 0.7+ | Enterprise-grade, free tier, HNSW |
| Cache | Upstash Redis (REST) | Latest | Serverless-compatible, free tier |
| Frontend | React 18 + Vite | React 18.3 | Fast dev, small bundle |
| Styling | Tailwind CSS | 3.4+ | Utility-first, RTL plugin |
| State/data fetching | TanStack Query | 5.x | Cache-aware query management |
| i18n | i18next + react-i18next | 23.x | EN/AR switching, RTL |
| Containerisation | Docker + docker-compose | Latest | Reproducible, deployable |
| CI/CD | GitHub Actions | Latest | Automatic build + deploy on push |
| Deployment (backend) | Railway | - | Free tier, Docker native, auto-deploy |
| Deployment (frontend) | Vercel | - | Free tier, Vite native, instant deploy |
| Database | Supabase (PostgreSQL + pgvector) | Latest | Free tier, managed, REST + realtime |
| Monitoring | Railway metrics + Supabase dashboard | - | Free, zero config |

---

## 17. Repository Structure

```
mumzsense/
│
├── backend/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── classifier_agent.py       # TF-IDF + BiLSTM ensemble (Gate A slot)
│   │   ├── rag_agent.py              # pgvector retrieval (Gate C + D slot)
│   │   ├── response_agent.py         # Llama synthesis
│   │   └── escalation_handler.py    # Gate B — typed stub in Phase 1
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py                  # AgentState TypedDict
│   │   └── pipeline.py              # LangGraph graph definition
│   │
│   ├── models/                       # Trained model artefacts (gitignored, mounted at runtime)
│   │   ├── tfidf_vectoriser.pkl
│   │   ├── logreg_topic.pkl
│   │   ├── logreg_urgency.pkl
│   │   ├── bilstm_weights.h5
│   │   ├── bilstm_tokeniser.pkl
│   │   ├── ensemble_weights.json
│   │   └── calibration_params.json
│   │
│   ├── data/
│   │   ├── generate_corpus.py        # Llama API corpus generation script
│   │   ├── verify_corpus.py          # Schema validation + quality checks
│   │   ├── eda.py                    # EDA analysis script
│   │   ├── preprocess.py             # Cleaning + feature engineering
│   │   ├── embed_and_index.py        # Embedding + Supabase ingestion
│   │   ├── corpus_validated.jsonl    # Generated corpus (gitignored — large)
│   │   └── eda_report.json           # EDA output (committed)
│   │
│   ├── training/
│   │   ├── train_classifier.py       # Full training pipeline
│   │   ├── evaluate.py               # Test set evaluation
│   │   └── optimise_ensemble.py     # Bayesian weight search
│   │
│   ├── cache/
│   │   └── redis_client.py           # Upstash Redis wrapper
│   │
│   ├── db/
│   │   ├── schema.sql                # PostgreSQL + pgvector schema
│   │   └── supabase_client.py       # Supabase connection
│   │
│   ├── schemas.py                    # Pydantic request/response models
│   ├── main.py                       # FastAPI app, routes, CORS, rate limiting
│   ├── config.py                     # Environment variable loading
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatThread.jsx
│   │   │   ├── MessageBubble.jsx
│   │   │   ├── SourceCard.jsx
│   │   │   ├── UncertaintyBanner.jsx
│   │   │   ├── UrgencyBadge.jsx
│   │   │   ├── StageSelector.jsx
│   │   │   ├── LanguageToggle.jsx
│   │   │   └── LoadingSkeleton.jsx
│   │   ├── screens/
│   │   │   ├── Landing.jsx
│   │   │   ├── Chat.jsx
│   │   │   └── Error.jsx
│   │   ├── hooks/
│   │   │   └── useQuery.js
│   │   ├── i18n/
│   │   │   ├── en.json
│   │   │   └── ar.json
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── index.html
│   ├── tailwind.config.js
│   ├── vite.config.js
│   └── package.json
│
├── evals/
│   ├── test_cases.json               # 20+ test cases with expected outputs
│   ├── run_evals.py                  # Evaluation harness
│   └── results.json                 # Latest eval run output (committed)
│
├── .github/
│   └── workflows/
│       ├── deploy_backend.yml        # Railway deploy on push to main
│       └── deploy_frontend.yml      # Vercel deploy on push to main
│
├── docker-compose.yml               # Local development orchestration
├── .env.example                     # Template — actual .env gitignored
├── README.md
├── EVALS.md                         # Honest evaluation writeup
└── TRADEOFFS.md                     # Architecture decisions + tradeoffs
```

---

## 18. Docker & Containerised Delivery

### 18.1 Backend Dockerfile

Multi-stage build to keep the final image small:

**Stage 1 — Builder:**
- Base: `python:3.11-slim`
- Install build dependencies (gcc, g++ for TF-IDF native extensions)
- Install all Python requirements
- Pre-download HuggingFace models to the image (bge-large, multilingual-mpnet) — avoids runtime download delays

**Stage 2 — Runner:**
- Base: `python:3.11-slim`
- Copy only the installed packages from builder (not build tools)
- Copy application code
- Copy pre-downloaded model cache from builder
- Mount `models/` directory as a volume (model artefacts loaded at runtime, not baked into image)
- Expose port 8000
- CMD: `uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2`

**Expected image size:** ~2.5GB (dominated by TF models — acceptable for Railway deployment)

### 18.2 docker-compose.yml (Local Development)

Services:
- `backend`: FastAPI app, builds from `backend/Dockerfile`, mounts `backend/models/` volume
- `redis`: Redis 7 Alpine (local dev only — production uses Upstash)
- `postgres`: PostgreSQL 16 + pgvector extension (local dev only — production uses Supabase)

Environment variables passed from `.env` file. All services on the same Docker network.

### 18.3 CI/CD Pipeline

**GitHub Actions — backend deploy to Railway:**
- Trigger: push to `main` branch, changes in `backend/` directory
- Steps: checkout → docker build → push to Railway (via Railway CLI) → health check

**GitHub Actions — frontend deploy to Vercel:**
- Trigger: push to `main`, changes in `frontend/` directory
- Steps: checkout → `vercel deploy --prod` via Vercel CLI

**Model artefacts:** Not stored in the repo (too large for Git). Stored in a private Supabase Storage bucket. The Railway deployment pulls the latest model artefacts from Supabase Storage on container startup via a startup script.

---

## 19. Free Deployment Strategy

### 19.1 Service Allocation

| Service | Provider | Free Tier Limits | Notes |
|---------|---------|-----------------|-------|
| Backend API | Railway | $5/month credit (≈ 500 req-hours) | Sufficient for demo + evaluation |
| Frontend | Vercel | Unlimited for personal projects | Zero cost |
| Database | Supabase | 500MB storage, 2GB bandwidth | Sufficient for 560-post corpus + demo traffic |
| Vector search | Supabase (pgvector) | Included | No additional cost |
| Cache | Upstash Redis | 10,000 commands/day | Sufficient for demo |
| LLM | Llama API | Free tier (rate limited) | Sufficient for demo + eval |
| Embeddings | HuggingFace models | Free, self-hosted | Run in backend container |
| CI/CD | GitHub Actions | 2,000 min/month | Sufficient |
| Model storage | Supabase Storage | 1GB free | Sufficient for classifier artefacts |

**Total cost: $0** during evaluation period.

### 19.2 Accelerators for Free LLM Inference

If the Llama API free tier rate limits are hit during intensive evaluation:

- **OpenRouter free models**: `meta-llama/llama-3.1-70b-instruct:free` — same model, different gateway, independent rate limit. Swap via environment variable, zero code change.
- **Groq free tier**: `llama-3.1-70b-versatile` — extremely fast inference (up to 300 tokens/second), generous free tier. P95 latency drops from ~3s to ~0.8s. The API is OpenAI-compatible, requiring only a base URL change.
- **Together AI free tier**: Additional fallback, same model family.

The backend implements a **LLM gateway abstraction** with an environment variable `LLM_PROVIDER` (values: `llama_api`, `openrouter`, `groq`). Switching provider is a single env var change with zero code modification. This is also the extension gate for using Anthropic Claude in Phase 2.

### 19.3 Embedding Acceleration

HuggingFace models run on Railway's CPU. Embedding 560 posts takes approximately 8 minutes on CPU at 2.5 GHz. This is a one-time cost at deploy time, not per-query. Per-query embedding (for the incoming user query) takes < 300ms on CPU — acceptable.

If Railway's CPU is insufficient: use HuggingFace Inference API (free tier, 1,000 requests/day) for embedding — same models, remote execution, zero infrastructure cost.

---

## 20. How Components Tie Together — End-to-End Flow

This section describes the exact execution path for a single user query from browser to response.

**Scenario:** A mother types "my 3 month old refuses to latch after 2 weeks of feeding well" in English, with stage "0-3m" selected.

1. **Frontend** captures the query string and stage hint. Sends `POST /query` to Railway backend URL via TanStack Query. Shows loading skeleton.

2. **FastAPI `/query` handler** receives and validates the request via Pydantic model. Passes to the LangGraph pipeline.

3. **`cache_check_node`**: Normalises the query string, computes SHA-256 hash (with stage_hint appended), queries Upstash Redis. MISS (first time this query is seen). Sets `state["cached"] = False`.

4. **`classifier_node`**: 
   - `langdetect` identifies English
   - TF-IDF vectoriser transforms the query text → 8004-dim feature vector
   - Logistic Regression predicts: topic="feeding" (P=0.91), urgency="monitor" (P=0.77)
   - BiLSTM tokenises and pads the sequence → 256-dim hidden state → predicts: topic="feeding" (P=0.88), urgency="monitor" (P=0.81)
   - Ensemble: topic_confidence = 0.55×0.91 + 0.45×0.88 = 0.896, urgency_confidence = 0.45×0.77 + 0.55×0.81 = 0.792
   - Calibration applied: final confidence = 0.87
   - defer_flag = False (confidence > 0.60, urgency != seek-help)
   - Logs BiLSTM hidden state to `feedback_log` table (Gate A)

5. **`defer_router`**: defer_flag is False → routes to `rag_node`

6. **`rag_node`**:
   - Embeds query with bge-large (EN): 1024-dim vector
   - Queries Supabase: `WHERE stage = '0-3m' AND topic = 'feeding' ORDER BY en_embedding <=> $1 LIMIT 10`
   - Post-filters to similarity > 0.45: returns 7 candidates
   - Returns top-5 with scores [0.84, 0.79, 0.74, 0.68, 0.61]
   - retrieval_confidence = "high" (max_score = 0.84)

7. **`threshold_router`**: retrieval_confidence != "none" → routes to `synthesis_node`

8. **`synthesis_node`**:
   - Constructs prompt with 5 retrieved posts (situation + advice + outcome for each)
   - Sends to Llama 3.1 70B via Llama API: temperature=0.4, max_tokens=600
   - LLM generates warm, peer-voiced response citing "Post 1" and "Post 3" as primary sources
   - Post-processing hallucination check: scans response for named entities not in retrieved posts — none found
   - Constructs bilingual response: EN primary, AR secondary (second Llama call with AR system prompt)

9. **`cache_write_node`**: Writes response to Upstash Redis with key `mumzsense:query:{hash}`, TTL = 24 hours (routine urgency).

10. **FastAPI** returns the response JSON. Total time: ~3.2 seconds (dominated by Llama API calls).

11. **Frontend** receives response. TanStack Query updates state. Chat thread renders: MumzMind bubble with warm answer, urgency badge ("Worth mentioning to your paediatrician"), two expandable SourceCards.

---

## 21. Next Steps to Final Product (Phase 2)

Phase 2 transforms MumzSense v1 from a sequential RAG pipeline into a full multi-agent triage system. Every Phase 2 addition uses an extension gate built into Phase 1 — no rewrites, only additions.

### Step 1 — Activate Gate B: Real Escalation Agent
- Build paediatric triage KB from WHO, NHS, AAP public guidelines (chunked, embedded, indexed in a second pgvector table)
- Replace `escalation_handler.py` stub with a real agent that queries the KB, constructs a structured triage card (`{severity, red_flags, green_flags, recommended_action, sources}`)
- Add a "doctor referral" flow: integrates with a directory of Mumzworld partner paediatricians

### Step 2 — Activate Gate C: LangGraph Supervisor
- Add `supervisor_node` that receives classifier output and routes to the most appropriate specialist agent (feeding specialist RAG, sleep specialist RAG, health/triage agent)
- Build specialist RAG pools: separate pgvector collections per topic, each with deeper domain coverage
- Add conditional edges in the LangGraph graph based on topic + urgency combination

### Step 3 — Activate Gate D: Hybrid GraphRAG
- Deploy Neo4j AuraDB free tier
- Ingest post corpus into Neo4j as nodes with SIMILAR_TO, LED_TO, and BELONGS_TO edges
- Implement dual retrieval in `rag_agent.py`: vector similarity + Cypher graph traversal
- Merge retrieval results with weighted scoring (vector × 0.6 + graph × 0.4)
- This enables outcome chain reasoning: "mothers with this feeding issue often encountered X at the next stage"

### Step 4 — Activate Gate A: MADRL Policy
- Attach PPO policy head to BiLSTM using `stable-baselines3`
- Define reward function: R = α·rating + β·grounding - γ·hallucination - δ·false_urgency
- Run policy update loop on accumulated `feedback_log` data (requires ~1,000 logged interactions before meaningful signal)
- Classifier progressively improves without manual retraining

### Step 5 — Real Community Data
- Build the community forum (React + Supabase Realtime for live posts)
- Implement consent architecture: account-level opt-in, post-level toggle, withdrawal triggers vector index deletion
- Run real posts through the authenticity pipeline (classifier scores → trust_score assignment → embed → index)
- At ~10,000 consented posts, the synthetic corpus becomes a training scaffold rather than the primary knowledge base

### Step 6 — Production Hardening
- Migrate from Railway free tier to a paid tier or AWS ECS
- Add LLM fallback chain: Llama API → Groq → OpenRouter (automatic failover)
- Add comprehensive observability: structured logging to Datadog or Grafana Cloud (free tiers)
- Add Arabic AR quality evaluation pipeline (automated + human-in-the-loop)
- Load test to 100 concurrent users, optimise pgvector HNSW parameters for larger corpus

---

*End of MumzSense Phase 1 PRD*

---

**Document metadata**
- Version: 1.0
- Status: Ready for implementation
- Estimated build time: 14–16 hours (1.5 days with focused execution)
- Open questions requiring decision before build starts:
  - Confirm Llama API key is provisioned and rate limits checked
  - Confirm Supabase project is created and pgvector extension enabled
  - Confirm Railway account created and CLI installed
  - Decide on AR dialect: Gulf (Khaleeji) vs Levantine — recommend Gulf for Mumzworld's GCC market
