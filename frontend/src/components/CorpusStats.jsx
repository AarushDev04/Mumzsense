import { useTranslation } from 'react-i18next'
import { useCorpusStats, useHealth } from '../hooks/useApi'

function StatPill({ label, value, color }) {
  return (
    <div className="flex items-center gap-2">
      <div className={`w-2 h-2 rounded-full ${color}`} />
      <span className="text-xs text-warmgray-500 font-body">{label}</span>
      <span className="text-xs font-medium text-warmgray-600 ml-auto">{value}</span>
    </div>
  )
}

const TOPIC_COLORS = {
  feeding: 'bg-orange-400',
  sleep: 'bg-indigo-400',
  health: 'bg-red-400',
  development: 'bg-green-400',
  gear: 'bg-blue-400',
  postpartum: 'bg-purple-400',
  mental_health: 'bg-pink-400',
}

export function CorpusStatsPanel() {
  const { t } = useTranslation()
  const { data: stats } = useCorpusStats()
  const { data: health } = useHealth()

  const fallbackStats = {
    total: 560,
    by_lang: { en: 440, ar: 120 },
    by_topic: { feeding: 135, sleep: 88, health: 114, development: 79, gear: 36, postpartum: 44, mental_health: 64 },
    by_urgency: { routine: 350, monitor: 95, 'seek-help': 115 },
  }

  const s = stats || fallbackStats

  return (
    <div className="rounded-2xl bg-warmgray-50 border border-warmgray-100 p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-warmgray-600 font-body">{t('aboutTitle')}</h3>
        <div className="flex items-center gap-1.5">
          <div className={`w-2 h-2 rounded-full animate-pulse ${health?.status === 'ok' ? 'bg-green-400' : 'bg-amber-400'}`} />
          <span className="text-xs text-warmgray-400">
            {health?.status === 'ok' ? t('systemOnline') : t('systemDegraded')}
          </span>
        </div>
      </div>

      {/* Total */}
      <div className="text-center py-2">
        <p className="text-3xl font-display font-bold text-coral-400">{s.total.toLocaleString()}+</p>
        <p className="text-xs text-warmgray-400 mt-0.5">{t('totalPosts')}</p>
      </div>

      {/* Language split bar */}
      <div>
        <div className="flex items-center gap-1 mb-1.5">
          <span className="text-xs text-warmgray-400 font-body">EN</span>
          <div className="flex-1 h-2 rounded-full overflow-hidden bg-warmgray-100 flex">
            <div
              className="h-full bg-coral-300 transition-all"
              style={{ width: `${(s.by_lang.en / s.total) * 100}%` }}
            />
            <div className="h-full bg-teal-DEFAULT flex-1" style={{ background: '#4da6a0' }} />
          </div>
          <span className="text-xs text-warmgray-400 font-body">AR</span>
        </div>
        <div className="flex justify-between text-xs text-warmgray-300">
          <span>{s.by_lang.en} EN</span>
          <span>{s.by_lang.ar} AR</span>
        </div>
      </div>

      {/* Topics */}
      <div className="space-y-1.5">
        {Object.entries(s.by_topic).map(([topic, count]) => (
          <StatPill
            key={topic}
            label={t(`topics.${topic}`)}
            value={count}
            color={TOPIC_COLORS[topic] || 'bg-warmgray-300'}
          />
        ))}
      </div>

      {/* Health details */}
      {health && (
        <div className="pt-2 border-t border-warmgray-100 space-y-1">
          <div className="flex justify-between text-xs text-warmgray-300">
            <span>DB</span>
            <span className={health.db_status === 'ok' ? 'text-green-500' : 'text-amber-500'}>
              {health.db_status}
            </span>
          </div>
          <div className="flex justify-between text-xs text-warmgray-300">
            <span>Cache</span>
            <span className={health.redis_status === 'ok' ? 'text-green-500' : 'text-warmgray-400'}>
              {health.redis_status}
            </span>
          </div>
          <div className="flex justify-between text-xs text-warmgray-300">
            <span>Classifier</span>
            <span className={health.model_loaded ? 'text-green-500' : 'text-amber-500'}>
              {health.model_loaded ? 'loaded' : 'not loaded'}
            </span>
          </div>
          <div className="flex justify-between text-xs text-warmgray-300">
            <span>Version</span>
            <span>{health.version}</span>
          </div>
        </div>
      )}
    </div>
  )
}
