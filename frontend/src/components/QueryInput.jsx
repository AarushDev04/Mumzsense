import { useState, useRef, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { useApp } from '../context/AppContext'

export function QueryInput({ onSubmit, isLoading, disabled }) {
  const { t, i18n } = useTranslation()
  const { language } = useApp()
  const [value, setValue] = useState('')
  const [error, setError] = useState('')
  const textareaRef = useRef(null)
  const isRtl = language === 'ar'

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [value])

  const validate = (v) => {
    if (v.trim().length < 3) return t('queryTooShort')
    if (v.trim().length > 1000) return t('queryTooLong')
    return ''
  }

  const handleSubmit = (e) => {
    e?.preventDefault()
    const err = validate(value)
    if (err) { setError(err); return }
    setError('')
    onSubmit(value.trim())
    setValue('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const charCount = value.length
  const overLimit = charCount > 1000

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div
        className={`
          flex items-end gap-3 rounded-2xl border bg-white shadow-sm px-4 py-3
          transition-all duration-200
          ${error ? 'border-red-300 shadow-red-50' : 'border-warmgray-200 focus-within:border-coral-300 focus-within:shadow-coral-50/50 focus-within:shadow-md'}
        `}
        dir={isRtl ? 'rtl' : 'ltr'}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={e => { setValue(e.target.value); setError('') }}
          onKeyDown={handleKeyDown}
          placeholder={t('placeholder')}
          disabled={isLoading || disabled}
          rows={1}
          className={`
            flex-1 resize-none bg-transparent text-warmgray-700 placeholder-warmgray-300
            text-sm leading-relaxed outline-none max-h-32 overflow-y-auto
            ${isRtl ? 'font-arabic text-right' : 'font-body'}
            disabled:opacity-50
          `}
          dir={isRtl ? 'rtl' : 'ltr'}
          aria-label={t('placeholder')}
        />

        <div className="flex items-center gap-2 flex-shrink-0">
          {/* Character count — only show near limit */}
          {charCount > 800 && (
            <span className={`text-xs ${overLimit ? 'text-red-500' : 'text-warmgray-300'}`}>
              {charCount}/1000
            </span>
          )}
          
          <button
            type="submit"
            disabled={isLoading || disabled || !value.trim() || overLimit}
            className={`
              flex items-center justify-center w-9 h-9 rounded-xl
              transition-all duration-150
              ${isLoading || !value.trim() || overLimit
                ? 'bg-warmgray-100 text-warmgray-300 cursor-not-allowed'
                : 'bg-coral-400 text-white hover:bg-coral-500 active:scale-95 shadow-sm'
              }
            `}
            aria-label={t('askButton')}
          >
            {isLoading ? (
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d={isRtl ? 'M15 19l-7-7 7-7' : 'M9 5l7 7-7 7'} />
              </svg>
            )}
          </button>
        </div>
      </div>

      {error && (
        <p className="mt-1.5 px-1 text-xs text-red-500 font-body animate-fade-in">
          {error}
        </p>
      )}
    </form>
  )
}
