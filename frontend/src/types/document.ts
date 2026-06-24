export interface DocumentMeta {
  id: string
  original_filename: string
  file_size_bytes: number | null
  mime_type: string | null
  is_processed: boolean
  created_at: string
}

export interface Job {
  id: string
  document_v1_id: string
  document_v2_id: string
  v1_filename: string
  v2_filename: string
  status: string
  current_phase: string
  progress_pct: number
  error_message: string | null
  total_pairs: number | null
  matched_count: number | null
  added_count: number | null
  deleted_count: number | null
  split_count: number | null
  merge_count: number | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}
