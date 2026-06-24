import { useEffect, useRef, useCallback } from 'react'
import { useAuthStore } from '@/stores/auth'
import type { WSProgressMessage } from '@/types/job'

type MessageHandler = (data: WSProgressMessage) => void

const WS_BASE = import.meta.env.VITE_WS_BASE_URL || '/ws'

export function useWebSocket(jobId: string | null, onMessage: MessageHandler) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectRef = useRef(0)
  const handlerRef = useRef(onMessage)

  handlerRef.current = onMessage

  const connect = useCallback(() => {
    if (!jobId) return

    const token = useAuthStore.getState().token
    const ws = new WebSocket(`${WS_BASE}/jobs/${jobId}?token=${encodeURIComponent(token || '')}`)
    wsRef.current = ws

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      handlerRef.current(data)
    }

    ws.onclose = () => {
      // Exponential backoff reconnect (max 30 seconds)
      const delay = Math.min(1000 * 2 ** reconnectRef.current, 30000)
      reconnectRef.current += 1
      setTimeout(() => connect(), delay)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [jobId])

  useEffect(() => {
    reconnectRef.current = 0
    connect()
    return () => {
      wsRef.current?.close()
    }
  }, [connect])

  return wsRef
}
