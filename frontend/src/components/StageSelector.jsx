import { useTranslation } from 'react-i18next'
import { useApp } from '../context/AppContext'

const STAGES = ['trimester', 'newborn', '0-3m', '3-6m', '6-12m', 'toddler']

const STAGE_EMOJIS = {
  trimester: '🤰',
  newborn: '👶',
  '0-3m': '🌸',
  '3-6m': '😊',
  '6-12m': '🍼',
  toddler: '🚶',
}

export function StageSelector() {
  const { t } = useTranslation()
  const { selectedStage, setSelectedStage } = useApp()

  return (
    <div className="space-y-2">
      <p className="text-xs font-medium text-warmgray-400 uppercase tracking-wider px-1">
        {t('stages.label')}
      </p>
      <div className="flex flex-wrap gap-2">
        {STAGES.map(stage => {
          const isSelected = selectedStage === stage
          return (
            <button
              key={stage}
              onClick={() => setSelectedStage(isSelected ? null : stage)}
              className={`
                flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium
                border transition-all duration-150
                ${isSelected
                  ? 'bg-coral-400 border-coral-400 text-white shadow-sm'
                  : 'bg-white border-warmgray-200 text-warmgray-500 hover:border-coral-300 hover:text-coral-500'
                }
              `}
            >
              <span className="text-base leading-none">{STAGE_EMOJIS[stage]}</span>
              <span className="font-body">{t(`stages.${stage}`)}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
