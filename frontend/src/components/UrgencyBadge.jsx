import { useTranslation } from 'react-i18next'

const URGENCY_STYLES = {
  routine: null, // no badge for routine
  monitor: {
    bg: 'bg-amber-50 border-amber-200',
    text: 'text-amber-700',
    dot: 'bg-amber-400',
    icon: '⚠',
  },
  'seek-help': {
    bg: 'bg-red-50 border-red-200',
    text: 'text-red-700',
    dot: 'bg-red-500',
    icon: '🏥',
  },
}

export function UrgencyBadge({ urgency }) {
  const { t } = useTranslation()
  const style = URGENCY_STYLES[urgency]
  if (!style) return null

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-medium ${style.bg} ${style.text}`}
    >
      <span className="text-sm">{style.icon}</span>
      {t(`urgency.${urgency}`)}
    </span>
  )
}

export function UrgencyBanner({ urgency, lang }) {
  const { t } = useTranslation()
  if (urgency !== 'seek-help') return null

  return (
    <div className="mt-3 rounded-xl bg-teal-50 border border-teal-200 p-4 flex items-start gap-3">
      <div className="text-2xl flex-shrink-0">🏥</div>
      <div>
        <p className="text-sm font-medium text-teal-900 mb-1">
          {t('seekHelpTitle')}
        </p>
        <a
          href="https://www.mumzworld.com/"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-sm text-teal-700 hover:text-teal-900 font-medium underline underline-offset-2 transition-colors"
        >
          {t('seekHelpCTA')} →
        </a>
      </div>
    </div>
  )
}
