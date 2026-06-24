import type { JobProgress, Phase } from '@/types/job'
import { PHASE_LABELS } from '@/types/job'

const PHASE_ORDER: Phase[] = ['phase1', 'phase2', 'phase3']

function getPhaseState(step: Phase, current: Phase): 'done' | 'active' | 'pending' {
  const stepIdx = PHASE_ORDER.indexOf(step)
  const currentIdx = PHASE_ORDER.indexOf(current)

  if (current === 'completed') return 'done'
  if (current === 'failed') return stepIdx < PHASE_ORDER.length ? 'done' : 'pending'
  if (stepIdx < currentIdx) return 'done'
  if (stepIdx === currentIdx) return 'active'
  return 'pending'
}

interface PhaseStepperProps {
  progress: JobProgress
}

export function PhaseStepper({ progress }: PhaseStepperProps) {
  return (
    <div className="space-y-4">
      {PHASE_ORDER.map((phase, idx) => {
        const state = getPhaseState(phase, progress.phase)
        const phaseInfo = progress.phases[idx]

        return (
          <div key={phase} className="flex items-start gap-3">
            {/* Step indicator */}
            <div className="mt-0.5 flex flex-col items-center">
              <div
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-medium
                  ${state === 'done' ? 'bg-diff-addition text-white' : ''}
                  ${state === 'active' ? 'bg-accent text-white' : ''}
                  ${state === 'pending' ? 'border-2 border-muted-foreground/25 text-muted-foreground' : ''}
                `}
              >
                {state === 'done' ? (
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  idx + 1
                )}
              </div>
              {idx < 2 && (
                <div
                  className={`mt-1 h-6 w-0.5 ${state === 'done' ? 'bg-diff-addition' : 'bg-muted-foreground/20'}`}
                />
              )}
            </div>

            {/* Phase label + progress */}
            <div className="min-w-0 flex-1">
              <p
                className={`text-sm font-medium ${
                  state === 'active' ? 'text-accent' : state === 'done' ? 'text-foreground' : 'text-muted-foreground'
                }`}
              >
                {PHASE_LABELS[phase]}
              </p>
              {phaseInfo?.message && (
                <p className="text-xs text-muted-foreground">{phaseInfo.message}</p>
              )}
              {state === 'active' && (
                <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-accent transition-all duration-500"
                    style={{ width: `${phaseInfo?.progress ?? 0}%` }}
                  />
                </div>
              )}
            </div>
          </div>
        )
      })}

      {progress.phase === 'completed' && (
        <div className="rounded-md bg-diff-addition/10 px-4 py-3 text-sm text-diff-addition">
          So sánh hoàn tất. Báo cáo đã sẵn sàng.
        </div>
      )}
      {progress.phase === 'failed' && (
        <div className="rounded-md bg-destructive/10 px-4 py-3 text-sm text-destructive">
          Đã xảy ra lỗi trong quá trình xử lý.
        </div>
      )}
    </div>
  )
}
