import { createContext, useContext, useState, useCallback } from 'react'
import i18next from 'i18next'
import { initReactI18next } from 'react-i18next'
import { resources, defaultNS, fallbackLng } from '../utils/i18n'

// Init i18next
i18next.use(initReactI18next).init({
  resources,
  defaultNS,
  fallbackLng,
  lng: 'en',
  interpolation: { escapeValue: false },
})

const AppContext = createContext(null)

export function AppProvider({ children }) {
  const [language, setLanguageState] = useState('en')
  const [selectedStage, setSelectedStage] = useState(null)
  const [chatHistory, setChatHistory] = useState([])
  const [systemStatus, setSystemStatus] = useState(null)

  const setLanguage = useCallback((lang) => {
    setLanguageState(lang)
    i18next.changeLanguage(lang)
    document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr'
    document.documentElement.lang = lang
  }, [])

  const addMessage = useCallback((message) => {
    setChatHistory(prev => [...prev, { ...message, id: Date.now() + Math.random() }])
  }, [])

  const clearHistory = useCallback(() => {
    setChatHistory([])
  }, [])

  return (
    <AppContext.Provider value={{
      language,
      setLanguage,
      selectedStage,
      setSelectedStage,
      chatHistory,
      addMessage,
      clearHistory,
      systemStatus,
      setSystemStatus,
    }}>
      {children}
    </AppContext.Provider>
  )
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used inside AppProvider')
  return ctx
}
