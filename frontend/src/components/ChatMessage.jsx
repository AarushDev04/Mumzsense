import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { CitationsPanel } from './SourceCard'
import { UrgencyBadge, UrgencyBanner } from './UrgencyBadge'
import { useFeedback } from '../hooks/useApi'
import { useApp } from '../context/AppContext'

// ── Typing indicator ─────────────────────────────────────────────────────────

export function TypingIndicator() {
  return (
    <div className="flex items-start gap-3 animate-fade-up">
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-coral-300 to-coral-400 flex-shrink-0 flex items-center justify-center text-white text-xs font-bold shadow-sm">
        M
      </div>
      <div className="bg-white border border-warmgray-100 rounded-2xl rounded-tl-none px-5 py-3.5 shadow-sm">
        <div className="flex items-center gap-1.5 h-5">
          {[0, 1, 2].map(i => (
            <div
              key={i}
              className="w-2 h-2 rounded-full bg-coral-300 animate-bounce-gentle"
              style={{ animationDelay: `${i * 0.16}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Loading skeleton ──────────────────────────────────────────────────────────

export function MessageSkeleton() {
  return (
    <div className="flex items-start gap-3 animate-fade-up">
      <div className="w-8 h-8 rounded-full bg-warmgray-100 flex-shrink-0 animate-pulse" />
      <div className="flex-1 space-y-2 max-w-lg">
        <div className="h-4 bg-warmgray-100 rounded-full w-3/4 animate-pulse" />
        <div className="h-4 bg-warmgray-100 rounded-full w-full animate-pulse" />
        <div className="h-4 bg-warmgray-100 rounded-full w-5/6 animate-pulse" />
        <div className="h-4 bg-warmgray-100 rounded-full w-2/3 animate-pulse" />
      </div>
    </div>
  )
}

// ── User message ──────────────────────────────────────────────────────────────

export function UserMessage({ content }) {
  const { language } = useApp()
  const isRtl = language === 'ar'

  return (
    <div className={`flex items-end gap-3 ${isRtl ? 'flex-row' : 'flex-row-reverse'} animate-fade-up`}>
      <div className="w-7 h-7 rounded-full bg-warmgray-200 flex-shrink-0" />
      <div
        className="max-w-xs md:max-w-md lg:max-w-lg px-4 py-3 rounded-2xl rounded-br-none bg-gradient-to-br from-coral-400 to-coral-500 text-white shadow-sm"
        dir={isRtl ? 'rtl' : 'ltr'}
      >
        <p className={`text-sm leading-relaxed ${isRtl ? 'font-arabic' : 'font-body'}`}>
          {content}
        </p>
      </div>
    </div>
  )
}

// ── Star rating ───────────────────────────────────────────────────────────────

function StarRating({ onRate }) {
  const { t } = useTranslation()
  const [rated, setRated] = useState(false)
  const [hovered, setHovered] = useState(0)
  const feedback = useFeedback()

  const handleRate = (stars) => {
    setRated(true)
    onRate?.(stars)
    feedback.mutate({
      queryHash: 'session-' + Date.now(),
      rating: stars,
      wasHelpful: stars >= 3,
      urgencyFelt: null,
    })
  }

  if (rated || feedback.isSuccess) {
    return (
      <p className="text-xs text-warmgray-400 mt-3">{t('feedbackThanks')}</p>
    )
  }

  return (
    <div className="flex items-center gap-2 mt-3">
      <span className="text-xs text-warmgray-400">{t('feedbackPrompt')}</span>
      <div className="flex gap-0.5">
        {[1, 2, 3, 4, 5].map(star => (
          <button
            key={star}
            onClick={() => handleRate(star)}
            onMouseEnter={() => setHovered(star)}
            onMouseLeave={() => setHovered(0)}
            className="text-lg transition-transform hover:scale-110"
          >
            <span className={(hovered || 0) >= star ? 'text-amber-400' : 'text-warmgray-200'}>
              ★
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Confidence chip ───────────────────────────────────────────────────────────

function ConfidenceChip({ level }) {
  const { t } = useTranslation()
  const STYLES = {
    high:      'text-green-600',
    medium:    'text-amber-600',
    low:       'text-orange-600',
    none:      'text-warmgray-400',
    deferred:  'text-gray-500',
  }
  const DOTS = {
    high: 'bg-green-400', medium: 'bg-amber-400', low: 'bg-orange-400',
    none: 'bg-warmgray-300', deferred: 'bg-gray-400',
  }
  return (
    <span className={`inline-flex items-center gap-1 text-xs ${STYLES[level] || STYLES.none}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${DOTS[level] || DOTS.none}`} />
      {t(`confidence.${level}`)}
    </span>
  )
}

// ── Assistant message ─────────────────────────────────────────────────────────

export function AssistantMessage({ message }) {
  const { t } = useTranslation()
  const { language } = useApp()
  const isRtl = language === 'ar'
  const [showSecondary, setShowSecondary] = useState(false)

  const {
    content,
    contentSecondary,
    citations = [],
    urgencyFlag,
    confidenceLevel,
    deferMessage,
    cached,
    latencyMs,
    hallucinationRisk,
  } = message

  // Uncertainty / defer state
  if (deferMessage && !content) {
    return (
      <div className="flex items-start gap-3 animate-fade-up">
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-warmgray-300 to-warmgray-400 flex-shrink-0 flex items-center justify-center text-white text-xs font-bold shadow-sm">
          M
        </div>
        <div className="max-w-lg bg-warmgray-50 border border-warmgray-200 rounded-2xl rounded-tl-none px-5 py-4 shadow-sm">
          <p className="text-sm font-medium text-warmgray-700 mb-1.5 font-body">
            {t('uncertaintyTitle')}
          </p>
          <p className="text-sm text-warmgray-500 font-body leading-relaxed">
            {t('uncertaintyBody')}
          </p>
        </div>
      </div>
    )
  }

  const displayContent = showSecondary && contentSecondary ? contentSecondary : content
  const displayIsAr = showSecondary ? language !== 'ar' : language === 'ar'

  return (
    <div className="flex items-start gap-3 animate-fade-up" dir={isRtl ? 'rtl' : 'ltr'}>
      {/* Avatar */}
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-coral-300 to-coral-500 flex-shrink-0 flex items-center justify-center text-white text-xs font-bold shadow-sm">
        M
      </div>

      <div className="flex-1 max-w-2xl">
        {/* Main bubble */}
        <div className="bg-white border border-warmgray-100 rounded-2xl rounded-tl-none px-5 py-4 shadow-sm">
          
          {/* Header row */}
          <div className="flex items-center gap-2 mb-3 flex-wrap">
            <span className="text-xs font-semibold text-warmgray-400 font-body tracking-wide uppercase">
              {t('answerBy')}
            </span>
            <UrgencyBadge urgency={urgencyFlag} />
            <ConfidenceChip level={confidenceLevel} />
            {cached && (
              <span className="text-xs text-teal-500 font-body">{t('cacheHit')}</span>
            )}
          </div>

          {/* Answer text */}
          <div
            className={`text-sm text-warmgray-700 leading-7 whitespace-pre-wrap ${displayIsAr ? 'font-arabic' : 'font-body'}`}
            dir={displayIsAr ? 'rtl' : 'ltr'}
          >
            {displayContent}
          </div>

          {/* Hallucination warning */}
          {hallucinationRisk && (
            <div className="mt-3 flex items-start gap-2 p-2.5 rounded-lg bg-amber-50 border border-amber-100">
              <span className="text-amber-500 flex-shrink-0">⚠️</span>
              <p className="text-xs text-amber-700 font-body">
                This response mentions specific names or details — please verify with a healthcare professional.
              </p>
            </div>
          )}

          {/* Language toggle */}
          {contentSecondary && (
            <button
              onClick={() => setShowSecondary(s => !s)}
              className="mt-3 text-xs text-warmgray-400 hover:text-coral-500 transition-colors font-body underline underline-offset-2"
            >
              {showSecondary ? t('viewInEnglish') : t('viewInArabic')}
            </button>
          )}

          {/* Seek-help CTA */}
          <UrgencyBanner urgency={urgencyFlag} />

          {/* Citations */}
          <CitationsPanel citations={citations} />

          {/* Meta + feedback */}
          <div className="mt-3 pt-3 border-t border-warmgray-50 flex items-center justify-between">
            <StarRating />
            {latencyMs > 0 && (
              <span className="text-xs text-warmgray-300 font-body">
                {latencyMs}ms
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Error message ─────────────────────────────────────────────────────────────

export function ErrorMessage({ onRetry }) {
  const { t } = useTranslation()
  return (
    <div className="flex items-start gap-3 animate-fade-up">
      <div className="w-8 h-8 rounded-full bg-red-100 flex-shrink-0 flex items-center justify-center text-red-500 text-sm">
        !
      </div>
      <div className="bg-red-50 border border-red-100 rounded-2xl rounded-tl-none px-5 py-4">
        <p className="text-sm font-medium text-red-700 mb-1">{t('errorTitle')}</p>
        <p className="text-sm text-red-500 mb-3">{t('errorBody')}</p>
        <button
          onClick={onRetry}
          className="text-sm text-red-600 hover:text-red-800 font-medium underline underline-offset-2 transition-colors"
        >
          {t('retry')}
        </button>
      </div>
    </div>
  )
}
