import type { WSProgressMessage } from '@/types/job'
import { PHASE_ORDER, PHASE_LABELS } from '@/types/job'

interface PhaseStepperProps {
  message: WSProgressMessage
}

function getPhaseState(
  stepIdx: number,
  currentPhase: string,
  isCompleted: boolean,
  isFailed: boolean,
): 'done' | 'active' | 'pending' {
  if (isCompleted) return 'done'
  if (isFailed) {
    // Show completed up to the current step, current step as failed
    const currentIdx = PHASE_ORDER.indexOf(currentPhase as (typeof PHASE_ORDER)[number])
    if (stepIdx < currentIdx) return 'done'
    if (stepIdx === currentIdx) return 'active'
    return 'pending'
  }
  const currentIdx = PHASE_ORDER.indexOf(currentPhase as (typeof PHASE_ORDER)[number])
  if (currentIdx === -1) return 'pending' // unknown phase
  if (stepIdx < currentIdx) return 'done'
  if (stepIdx === currentIdx) return 'active'
  return 'pending'
}

export function PhaseStepper({ message }: PhaseStepperProps) {
  const isCompleted = message.event === 'completed'
  const isFailed = message.event === 'error'
  const currentPhase = message.current_phase

  return (
    <div className="space-y-4">
      {PHASE_ORDER.map((phase, idx) => {
        const state = getPhaseState(idx, currentPhase, isCompleted, isFailed)

        return (
          <div key={phase} className="flex items-start gap-3">
            {/* Step indicator */}
            <div className="mt-0.5 flex flex-col items-center">
              <div
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-medium
                  ${state === 'done' ? 'bg-diff-addition text-white' : ''}
                  ${state === 'active' && !isFailed ? 'bg-accent text-white' : ''}
                  ${state === 'active' && isFailed ? 'bg-destructive text-white' : ''}
                  ${state === 'pending' ? 'border-2 border-muted-foreground/25 text-muted-foreground' : ''}
                `}
              >
                {state === 'done' ? (
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                ) : state === 'active' && isFailed ? (
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                ) : (
                  idx + 1
                )}
              </div>
              {idx < PHASE_ORDER.length - 1 && (
                <div
                  className={`mt-1 h-6 w-0.5 ${state === 'done' ? 'bg-diff-addition' : 'bg-muted-foreground/20'}`}
                />
              )}
            </div>

            {/* Phase label + progress */}
            <div className="min-w-0 flex-1">
              <p
                className={`text-sm font-medium ${
                  state === 'active' && isFailed
                    ? 'text-destructive'
                    : state === 'active'
                      ? 'text-accent'
                      : state === 'done'
                        ? 'text-foreground'
                        : 'text-muted-foreground'
                }`}
              >
                {PHASE_LABELS[phase]}
              </p>
              {state === 'active' && (
                <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${isFailed ? 'bg-destructive' : 'bg-accent'}`}
                    style={{ width: `${message.progress_pct}%` }}
                  />
                </div>
              )}
              {isFailed && state === 'active' && message.error && (
                <p className="mt-1 text-xs text-destructive">{message.error}</p>
              )}
            </div>
          </div>
        )
      })}

      {isCompleted && (
        <div className="rounded-md bg-diff-addition/10 px-4 py-3 text-sm text-diff-addition">
          So sánh hoàn tất. Báo cáo đã sẵn sàng.
        </div>
      )}
      {isFailed && (
        <div className="rounded-md bg-destructive/10 px-4 py-3 text-sm text-destructive">
          Đã xảy ra lỗi trong quá trình xử lý.
          {message.error && (
            <p className="mt-1 text-xs opacity-80">{message.error}</p>
          )}
        </div>
      )}
    </div>
  )
}
