import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useApp } from './context/AppContext'
import { useSubmitQuery } from './hooks/useApi'
import { QueryInput } from './components/QueryInput'
import { StageSelector } from './components/StageSelector'
import { CorpusStatsPanel } from './components/CorpusStats'
import {
  UserMessage, AssistantMessage, TypingIndicator, ErrorMessage
} from './components/ChatMessage'

// ── Landing screen (no history) ───────────────────────────────────────────────

function LandingHero({ onAsk, isLoading }) {
  const { t } = useTranslation()
  const { language, setLanguage } = useApp()
  const isRtl = language === 'ar'

  const EXAMPLE_QUERIES = {
    en: [
      "My 3-month-old won't sleep longer than 2 hours at night",
      "Is it normal for a newborn to lose weight in the first week?",
      "When should I introduce solid foods?",
      "I'm feeling overwhelmed and anxious since giving birth",
    ],
    ar: [
      "طفلي عمره 3 أشهر لا ينام أكثر من ساعتين في الليل",
      "هل من الطبيعي أن يفقد المولود الجديد وزنه في الأسبوع الأول؟",
      "متى يمكنني البدء في تقديم الطعام الصلب؟",
      "أشعر بالإرهاق والقلق منذ الولادة",
    ],
  }

  const examples = EXAMPLE_QUERIES[language] || EXAMPLE_QUERIES.en

  return (
    <div className={`flex flex-col items-center text-center max-w-2xl mx-auto px-4 pt-8 pb-4 ${isRtl ? 'font-arabic' : ''}`}>
      {/* Logo wordmark */}
      <div className="mb-8">
        <div className="inline-flex items-center gap-2 mb-1">
          <span className="text-3xl">🌸</span>
          <h1 className="text-2xl font-display font-bold text-warmgray-700">
            Mumz<span className="text-coral-400">Sense</span>
          </h1>
        </div>
        <p className="text-xs text-warmgray-300 font-body tracking-widest uppercase">by MumzMind</p>
      </div>

      {/* Headline */}
      <h2 className={`text-3xl md:text-4xl font-display font-bold text-warmgray-700 leading-tight mb-3 ${isRtl ? 'text-right w-full' : ''}`}>
        {t('tagline')}
      </h2>
      <p className={`text-base text-warmgray-400 mb-8 font-body leading-relaxed ${isRtl ? 'text-right w-full' : ''}`}>
        {t('subheadline')}
      </p>

      {/* Stage selector */}
      <div className={`w-full mb-6 ${isRtl ? 'text-right' : ''}`}>
        <StageSelector />
      </div>

      {/* Input */}
      <div className="w-full mb-6">
        <QueryInput onSubmit={onAsk} isLoading={isLoading} />
      </div>

      {/* Example queries */}
      <div className={`w-full space-y-2 mb-8 ${isRtl ? 'text-right' : ''}`}>
        <p className="text-xs text-warmgray-300 font-body uppercase tracking-wide">Try asking</p>
        <div className="flex flex-wrap gap-2 justify-center">
          {examples.map((q, i) => (
            <button
              key={i}
              onClick={() => onAsk(q)}
              disabled={isLoading}
              className="text-xs px-3 py-1.5 rounded-full border border-warmgray-200 text-warmgray-500 hover:border-coral-300 hover:text-coral-500 transition-all font-body text-left disabled:opacity-50"
              dir={isRtl ? 'rtl' : 'ltr'}
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Trust signal */}
      <p className="text-xs text-warmgray-300 font-body">
        ✨ {t('trustSignal')}
      </p>
    </div>
  )
}

// ── Chat thread view ──────────────────────────────────────────────────────────

function ChatThread({ messages, isLoading, onRetry, onReset }) {
  const { t } = useTranslation()
  const { language, setLanguage, selectedStage } = useApp()
  const { ask, isError, reset } = useSubmitQuery()
  const bottomRef = useRef(null)
  const isRtl = language === 'ar'

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const handleAsk = (q) => {
    reset()
    ask(q)
  }

  return (
    <div className="flex flex-col h-full" dir={isRtl ? 'rtl' : 'ltr'}>
      {/* Thread header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-warmgray-100 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <button
          onClick={onReset}
          className="flex items-center gap-2 text-sm text-warmgray-400 hover:text-warmgray-600 transition-colors font-body"
        >
          <span className="text-lg">🌸</span>
          <span className="font-display font-semibold text-warmgray-600">
            Mumz<span className="text-coral-400">Sense</span>
          </span>
        </button>
        <div className="flex items-center gap-3">
          {selectedStage && (
            <span className="text-xs px-2.5 py-1 rounded-full bg-coral-50 text-coral-400 border border-coral-100 font-body">
              {t(`stages.${selectedStage}`)}
            </span>
          )}
          <button
            onClick={() => setLanguage(language === 'en' ? 'ar' : 'en')}
            className="text-xs px-3 py-1.5 rounded-full border border-warmgray-200 text-warmgray-500 hover:border-coral-300 hover:text-coral-500 transition-all font-body"
          >
            {t('languageToggle')}
          </button>
        </div>
      </div>

      {/* Message list */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        {messages.map(msg => (
          <div key={msg.id}>
            {msg.role === 'user' ? (
              <UserMessage content={msg.content} />
            ) : (
              <AssistantMessage message={msg} />
            )}
          </div>
        ))}

        {isLoading && (
          <div>
            <TypingIndicator />
          </div>
        )}

        {isError && (
          <div>
            <ErrorMessage onRetry={() => { reset(); }} />
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input sticky at bottom */}
      <div className="border-t border-warmgray-100 bg-white/90 backdrop-blur-sm p-4">
        <StageSelector />
        <div className="mt-3">
          <QueryInput onSubmit={handleAsk} isLoading={isLoading} />
        </div>
      </div>
    </div>
  )
}

// ── Root App ──────────────────────────────────────────────────────────────────

export default function App() {
  const { language, setLanguage, chatHistory, addMessage } = useApp()
  const { ask, isLoading, isError, reset } = useSubmitQuery()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const isRtl = language === 'ar'
  const hasChatHistory = chatHistory.length > 0

  const handleAsk = (query) => {
    reset()
    ask(query)
  }

  const handleReset = () => {
    // Clear handled by page refresh — keeps it simple per PRD (no persistence)
    window.location.reload()
  }

  return (
    <div
      className="min-h-screen bg-warmgray-50 flex"
      dir={isRtl ? 'rtl' : 'ltr'}
    >
      {/* Sidebar — corpus stats */}
      <aside
        className={`
          hidden lg:flex flex-col w-72 flex-shrink-0 border-r border-warmgray-100 bg-white
          ${isRtl ? 'border-l border-r-0' : ''}
        `}
      >
        {/* Sidebar header */}
        <div className="p-4 border-b border-warmgray-100">
          <div className="flex items-center gap-2">
            <span className="text-xl">🌸</span>
            <div>
              <h1 className="text-base font-display font-bold text-warmgray-700">
                Mumz<span className="text-coral-400">Sense</span>
              </h1>
              <p className="text-xs text-warmgray-300 font-body">by MumzMind</p>
            </div>
          </div>
        </div>

        {/* Language toggle */}
        <div className="px-4 pt-4">
          <div className="flex rounded-xl border border-warmgray-100 overflow-hidden">
            {['en', 'ar'].map(lang => (
              <button
                key={lang}
                onClick={() => setLanguage(lang)}
                className={`
                  flex-1 py-2 text-sm font-medium transition-all font-body
                  ${language === lang
                    ? 'bg-coral-400 text-white'
                    : 'text-warmgray-400 hover:text-warmgray-600 hover:bg-warmgray-50'
                  }
                `}
              >
                {lang === 'en' ? 'English' : 'العربية'}
              </button>
            ))}
          </div>
        </div>

        {/* Stats */}
        <div className="flex-1 p-4 overflow-y-auto">
          <CorpusStatsPanel />
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-warmgray-100">
          <p className="text-xs text-warmgray-300 font-body text-center">
            MumzSense v1.0 · Phase 1
          </p>
          <p className="text-xs text-warmgray-200 font-body text-center mt-0.5">
            Powered by Llama 3.1 70B + RAG
          </p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col min-h-screen overflow-hidden">
        {hasChatHistory ? (
          <ChatThread
            messages={chatHistory}
            isLoading={isLoading}
            isError={isError}
            onRetry={() => reset()}
            onReset={handleReset}
          />
        ) : (
          <div className="flex-1 overflow-y-auto">
            {/* Mobile header */}
            <div className="lg:hidden flex items-center justify-between px-4 py-3 border-b border-warmgray-100 bg-white">
              <div className="flex items-center gap-2">
                <span className="text-lg">🌸</span>
                <span className="font-display font-bold text-warmgray-700 text-base">
                  Mumz<span className="text-coral-400">Sense</span>
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setLanguage(language === 'en' ? 'ar' : 'en')}
                  className="text-xs px-3 py-1.5 rounded-full border border-warmgray-200 text-warmgray-500 font-body"
                >
                  {language === 'en' ? 'عربي' : 'English'}
                </button>
              </div>
            </div>

            <LandingHero onAsk={handleAsk} isLoading={isLoading} />

            {isError && (
              <div className="max-w-xl mx-auto px-4 pb-4">
                <div className="p-4 rounded-xl bg-red-50 border border-red-100 text-sm text-red-600 text-center font-body">
                  Something went wrong. Please check your connection and try again.
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
