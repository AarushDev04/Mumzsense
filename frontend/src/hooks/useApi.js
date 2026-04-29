import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { submitQuery, checkHealth, getCorpusStats, submitFeedback } from '../utils/api'
import { useApp } from '../context/AppContext'
import { useCallback } from 'react'

/**
 * Hook: useQuery — submits a query to the RAG pipeline
 * Connects to POST /query
 */
export function useSubmitQuery() {
  const { language, selectedStage, addMessage } = useApp()
  const queryClient = useQueryClient()

  const mutation = useMutation({
    mutationFn: ({ query }) =>
      submitQuery(query, selectedStage, language),
    onSuccess: (data, variables) => {
      addMessage({
        role: 'user',
        content: variables.query,
      })
      addMessage({
        role: 'assistant',
        content: data.answer_primary,
        contentSecondary: data.answer_secondary,
        citations: data.citations || [],
        urgencyFlag: data.urgency_flag,
        confidenceLevel: data.confidence_level,
        deferMessage: data.defer_message,
        hallucinationRisk: data.hallucination_risk,
        cached: data.cached,
        latencyMs: data.latency_ms,
        queryHash: null, // will be generated server-side for feedback
      })
    },
  })

  const ask = useCallback((query) => {
    if (!query || query.trim().length < 3) return
    mutation.mutate({ query: query.trim() })
  }, [mutation])

  return {
    ask,
    isLoading: mutation.isPending,
    isError: mutation.isError,
    error: mutation.error,
    reset: mutation.reset,
  }
}

/**
 * Hook: useHealth — polls /health endpoint
 */
export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: checkHealth,
    refetchInterval: 60000, // poll every minute
    staleTime: 30000,
    retry: 2,
  })
}

/**
 * Hook: useCorpusStats — fetches /corpus/stats
 */
export function useCorpusStats() {
  return useQuery({
    queryKey: ['corpus-stats'],
    queryFn: getCorpusStats,
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: 1,
  })
}

/**
 * Hook: useFeedback — submits feedback to /feedback
 */
export function useFeedback() {
  return useMutation({
    mutationFn: ({ queryHash, rating, wasHelpful, urgencyFelt }) =>
      submitFeedback(queryHash, rating, wasHelpful, urgencyFelt),
  })
}
