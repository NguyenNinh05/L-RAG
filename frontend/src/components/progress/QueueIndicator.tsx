interface QueueIndicatorProps {
  position?: number
  estimatedWaitMinutes?: number
}

export function QueueIndicator({ position, estimatedWaitMinutes }: QueueIndicatorProps) {
  if (!position || position < 1) return null

  return (
    <div className="rounded-md border bg-muted/50 px-4 py-3">
      <div className="flex items-center gap-2">
        <div className="h-2 w-2 animate-pulse rounded-full bg-accent" />
        <p className="text-sm font-medium text-foreground">
          Đang chờ GPU
        </p>
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        Vị trí hàng đợi: <span className="tabular-nums font-medium">{position}</span>
        {estimatedWaitMinutes != null && (
          <> &middot; Thời gian chờ ước tính: ~{estimatedWaitMinutes} phút</>
        )}
      </p>
    </div>
  )
}
