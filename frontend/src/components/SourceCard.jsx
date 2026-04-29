import { useState } from 'react'
import { useTranslation } from 'react-i18next'

const TOPIC_COLORS = {
  feeding: { bg: 'bg-orange-50', border: 'border-orange-200', text: 'text-orange-700', dot: 'bg-orange-400' },
  sleep: { bg: 'bg-indigo-50', border: 'border-indigo-200', text: 'text-indigo-700', dot: 'bg-indigo-400' },
  health: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', dot: 'bg-red-400' },
  development: { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-700', dot: 'bg-green-400' },
  gear: { bg: 'bg-blue-50', border: 'border-blue-200', text: 'text-blue-700', dot: 'bg-blue-400' },
  postpartum: { bg: 'bg-purple-50', border: 'border-purple-200', text: 'text-purple-700', dot: 'bg-purple-400' },
  mental_health: { bg: 'bg-pink-50', border: 'border-pink-200', text: 'text-pink-700', dot: 'bg-pink-400' },
}

function TrustBar({ score, label }) {
  const pct = Math.round(score * 100)
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-warmgray-400 w-12 flex-shrink-0">{label}</span>
      <div className="flex-1 bg-warmgray-100 rounded-full h-1.5">
        <div
          className="h-1.5 rounded-full bg-gradient-to-r from-coral-300 to-coral-400 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-warmgray-400 w-8 text-right">{pct}%</span>
    </div>
  )
}

export function SourceCard({ post, index }) {
  const { t } = useTranslation()
  const [expanded, setExpanded] = useState(false)
  const colors = TOPIC_COLORS[post.topic] || TOPIC_COLORS.health

  const isArabic = post.lang === 'ar'

  return (
    <div
      className={`rounded-xl border ${colors.border} ${colors.bg} overflow-hidden transition-all duration-200`}
      dir={isArabic ? 'rtl' : 'ltr'}
    >
      {/* Header */}
      <button
        onClick={() => setExpanded(e => !e)}
        className="w-full px-4 py-3 flex items-center gap-3 text-left hover:opacity-80 transition-opacity"
      >
        <span className="text-lg font-display text-warmgray-300 flex-shrink-0">
          {String(index + 1).padStart(2, '0')}
        </span>
        
        <div className="flex-1 min-w-0">
          <p className={`text-xs font-medium ${colors.text} truncate`}>
            {t(`topics.${post.topic}`)} · {t(`stages.${post.stage}`)}
            {post.lang === 'ar' && (
              <span className="ml-1.5 inline-block px-1.5 py-0.5 rounded bg-white/60 text-warmgray-500 text-xs font-body">AR</span>
            )}
          </p>
          <p className="text-sm text-warmgray-500 truncate mt-0.5 font-body">
            {post.situation}
          </p>
        </div>

        <svg
          className={`w-4 h-4 text-warmgray-400 flex-shrink-0 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="px-4 pb-4 pt-0 animate-fade-in">
          <div className="h-px bg-warmgray-100 mb-3" />
          
          <div className="space-y-3">
            <div>
              <p className="text-xs font-medium text-warmgray-400 mb-1 uppercase tracking-wide">Situation</p>
              <p className="text-sm text-warmgray-600 font-body leading-relaxed">{post.situation}</p>
            </div>
            
            <div>
              <p className="text-xs font-medium text-warmgray-400 mb-1 uppercase tracking-wide">Advice</p>
              <p className="text-sm text-warmgray-700 font-body leading-relaxed font-medium">{post.advice}</p>
            </div>
            
            {post.outcome && (
              <div>
                <p className="text-xs font-medium text-warmgray-400 mb-1 uppercase tracking-wide">Outcome</p>
                <p className="text-sm text-warmgray-500 font-body leading-relaxed italic">{post.outcome}</p>
              </div>
            )}
            
            <div className="pt-1 space-y-1.5">
              <TrustBar score={post.trust_score} label={t('trustScore')} />
              <TrustBar score={post.similarity_score} label={t('similarity')} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export function CitationsPanel({ citations }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)

  if (!citations || citations.length === 0) return null

  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-2 text-sm text-warmgray-400 hover:text-warmgray-600 transition-colors"
      >
        <div className="flex -space-x-1">
          {citations.slice(0, 3).map((_, i) => (
            <div key={i} className="w-5 h-5 rounded-full bg-coral-100 border-2 border-white flex items-center justify-center">
              <span className="text-coral-500 text-xs font-bold">{i + 1}</span>
            </div>
          ))}
        </div>
        <span>
          {t('sources', { count: citations.length })}
        </span>
        <svg
          className={`w-3.5 h-3.5 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="mt-3 space-y-2 animate-fade-up">
          {citations.map((post, i) => (
            <SourceCard key={post.post_id || i} post={post} index={i} />
          ))}
        </div>
      )}
    </div>
  )
}
