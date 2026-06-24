import type { DocumentMeta } from '@/types/document'

export async function uploadDocument(file: File): Promise<DocumentMeta> {
  const formData = new FormData()
  formData.append('file', file)

  const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'
  const token = localStorage.getItem('legaldiff-auth')

  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  })

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body?.detail || `Upload failed: ${res.status}`)
  }

  return res.json()
}
