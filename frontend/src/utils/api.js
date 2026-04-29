/**
 * MumzSense API Service
 * Connects to all FastAPI backend endpoints:
 *   POST /query          — Main RAG pipeline
 *   GET  /health         — System health check
 *   GET  /corpus/stats   — Corpus statistics
 *   POST /feedback       — User feedback logging
 *   POST /admin/cache/flush  — Cache management (admin)
 *   GET  /evals/latest   — Latest eval results
 */

const BASE_URL = 'http://localhost:8001'

class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`
  let response
  try {
    response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    })
  } catch (networkError) {
    // Network-level failure (backend unreachable, CORS, etc.)
    throw new ApiError(
      `Network error: ${networkError.message || 'Unable to reach the server'}`,
      0,
      networkError.message
    )
  }

  if (!response.ok) {
    let detail = ''
    try {
      const contentType = response.headers.get('content-type') || ''
      if (contentType.includes('application/json')) {
        const err = await response.json()
        detail = err.detail || err.message || JSON.stringify(err)
      } else {
        const text = (await response.text()).trim()
        // Strip HTML tags if the backend returned an HTML error page
        detail = text.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 200)
      }
    } catch (_) {}
    throw new ApiError(
      `API error ${response.status}: ${detail || response.statusText}`,
      response.status,
      detail
    )
  }

  // Success path — guard against non-JSON content type
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    try {
      return await response.json()
    } catch (parseError) {
      // JSON parse failure on a 2xx response — surface a clear error
      throw new ApiError(
        `Received invalid JSON from server (${parseError.message})`,
        response.status,
        parseError.message
      )
    }
  }
  return response.text()
}

/**
 * POST /query
 * Main query endpoint — runs the full LangGraph RAG pipeline
 * @param {string} query - User's natural language question
 * @param {string|null} stageHint - Baby stage (trimester|newborn|0-3m|3-6m|6-12m|toddler)
 * @param {string|null} langPreference - Language override: 'en' | 'ar'
 * @returns {Promise<QueryResponse>}
 */
export async function submitQuery(query, stageHint = null, langPreference = null) {
  return request('/query', {
    method: 'POST',
    body: JSON.stringify({
      query,
      stage_hint: stageHint || undefined,
      lang_preference: langPreference || undefined,
    }),
  })
}

/**
 * GET /health
 * System health check — db, redis, model status
 * @returns {Promise<HealthResponse>}
 */
export async function checkHealth() {
  return request('/health')
}

/**
 * GET /corpus/stats
 * Returns corpus distribution statistics for the About section
 * @returns {Promise<CorpusStatsResponse>}
 */
export async function getCorpusStats() {
  return request('/corpus/stats')
}

/**
 * POST /feedback
 * Log user rating for MADRL Gate A preparation
 * @param {string} queryHash - SHA-256 hash of the query (returned in response)
 * @param {number} rating - 1–5 star rating
 * @param {boolean} wasHelpful - Was the answer helpful
 * @param {string|null} urgencyFelt - User's perceived urgency
 * @returns {Promise<FeedbackResponse>}
 */
export async function submitFeedback(queryHash, rating, wasHelpful, urgencyFelt = null) {
  return request('/feedback', {
    method: 'POST',
    body: JSON.stringify({
      query_hash: queryHash,
      rating,
      was_helpful: wasHelpful,
      urgency_felt: urgencyFelt || undefined,
    }),
  })
}

/**
 * GET /evals/latest
 * Fetch latest evaluation run results
 * @returns {Promise<object>}
 */
export async function getLatestEvals() {
  return request('/evals/latest')
}

/**
 * POST /admin/cache/flush
 * Flush Redis cache (requires admin token)
 * @param {string} adminToken
 * @returns {Promise<CacheFlushResponse>}
 */
export async function flushCache(adminToken) {
  return request('/admin/cache/flush', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${adminToken}`,
    },
  })
}

// Type documentation (for IDE support)
/**
 * @typedef {Object} QueryResponse
 * @property {string} answer_primary - Answer in query language
 * @property {string|null} answer_secondary - Answer in other language
 * @property {CitedPost[]} citations - Source posts cited
 * @property {string} urgency_flag - 'routine' | 'monitor' | 'seek-help'
 * @property {string} confidence_level - 'high' | 'medium' | 'low' | 'none'
 * @property {string|null} defer_message - Populated when pipeline deferred
 * @property {boolean} hallucination_risk - True if hallucination guard flagged
 * @property {boolean} cached - True if served from Redis cache
 * @property {number} latency_ms - Total pipeline latency in ms
 */

/**
 * @typedef {Object} CitedPost
 * @property {string} post_id
 * @property {string} situation
 * @property {string} advice
 * @property {string|null} outcome
 * @property {number} trust_score - 0–1
 * @property {number} similarity_score - Cosine similarity 0–1
 * @property {string} stage
 * @property {string} topic
 * @property {string} lang - 'en' | 'ar'
 */