export interface DocumentMeta {
  id: string
  filename: string
  file_size: number
  mime_type: string
  uploaded_at: string
  status: 'ready' | 'processing' | 'error'
}

export interface Job {
  id: string
  name: string
  doc_v1_id: string
  doc_v2_id: string
  status: 'pending' | 'phase1' | 'phase2' | 'phase3' | 'completed' | 'failed'
  created_at: string
}
