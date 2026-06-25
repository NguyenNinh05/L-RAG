// Benchmark evaluation runs — shapes mirror backend schemas/eval.py.

import type { AcuItem } from '@/types/diff'

export interface CategoryStat {
  gt_total: number
  detected: number
  recall?: number
}

export interface EvalRunSummary {
  slug: string
  label: string
  num_pairs: number
  change_f1: number
  change_f1_semantic: number
  change_recall: number
  change_precision: number
  hallucination_rate: number
  avg_time_per_pair_seconds: number
  total_acus_generated: number
  total_acus_passed: number
  total_acus_rejected: number
  per_category: Record<string, CategoryStat>
  per_category_semantic: Record<string, CategoryStat>
}

export interface EvalRunDetail {
  slug: string
  label: string
  num_pairs: number
  aggregate: Record<string, unknown>
  markdown_report: string | null
}

export interface EvalPairSummary {
  pair_name: string
  recall: number
  precision: number
  f1: number
  recall_semantic: number
  recall_article: number
  hallucination_rate: number
  total_time: number
  num_acus: number
}

// Reuse the report ACU shape.
export interface EvalPairDetail {
  pair_name: string
  v1_file: string | null
  v2_file: string | null
  gt_file: string | null
  phase1: Record<string, number>
  phase2: Record<string, number>
  phase3: Record<string, number>
  ground_truth: Record<string, unknown>
  change_detection: Record<string, number>
  per_category: Record<string, CategoryStat>
  per_category_semantic: Record<string, CategoryStat>
  acus: AcuItem[]
  total_time: number
  errors: unknown[]
}
