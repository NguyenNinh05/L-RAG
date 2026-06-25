import { api } from '@/api/client'
import type {
  EvalRunSummary,
  EvalRunDetail,
  EvalPairSummary,
  EvalPairDetail,
} from '@/types/eval'

export function listEvalRuns(): Promise<EvalRunSummary[]> {
  return api.get<EvalRunSummary[]>('/eval/runs')
}

export function getEvalRun(slug: string): Promise<EvalRunDetail> {
  return api.get<EvalRunDetail>(`/eval/runs/${slug}`)
}

export function listEvalPairs(slug: string): Promise<EvalPairSummary[]> {
  return api.get<EvalPairSummary[]>(`/eval/runs/${slug}/pairs`)
}

export function getEvalPair(slug: string, pairName: string): Promise<EvalPairDetail> {
  return api.get<EvalPairDetail>(`/eval/runs/${slug}/pairs/${encodeURIComponent(pairName)}`)
}
