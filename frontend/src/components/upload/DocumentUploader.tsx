import { useState, useRef, useCallback, type DragEvent } from 'react'

const ACCEPTED = ['.pdf', '.docx', 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
const MAX_SIZE = 50 * 1024 * 1024 // 50MB

const ACCEPTED_LABELS: Record<string, string> = {
  'application/pdf': 'PDF',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DOCX',
}

interface UploadedFile {
  id: string
  file: File
  progress: number
}

interface DocumentUploaderProps {
  onUpload: (file: File) => Promise<void>
  label?: string
  disabled?: boolean
}

export function DocumentUploader({ onUpload, label, disabled }: DocumentUploaderProps) {
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState<UploadedFile | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const validate = useCallback((file: File): string | null => {
    const ext = file.name.split('.').pop()?.toLowerCase()
    if (!ext || !['pdf', 'docx'].includes(ext)) {
      return 'Chỉ chấp nhận file PDF hoặc DOCX'
    }
    if (file.size > MAX_SIZE) {
      return 'File không được vượt quá 50MB'
    }
    return null
  }, [])

  const handleFile = async (file: File) => {
    const err = validate(file)
    if (err) {
      setError(err)
      return
    }
    setError(null)
    setUploading({ id: crypto.randomUUID(), file, progress: 0 })

    // Simulate progress
    const interval = setInterval(() => {
      setUploading((prev) => {
        if (!prev) return null
        const next = Math.min(prev.progress + 0.15, 0.9)
        return { ...prev, progress: next }
      })
    }, 100)

    try {
      await onUpload(file)
      clearInterval(interval)
      setUploading((prev) => prev ? { ...prev, progress: 1 } : null)
      // Clear after short delay
      setTimeout(() => setUploading(null), 800)
    } catch {
      clearInterval(interval)
      setUploading(null)
      setError('Upload thất bại. Vui lòng thử lại.')
    }
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const onDragOver = (e: DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }

  const onInputChange = () => {
    const file = inputRef.current?.files?.[0]
    if (file) handleFile(file)
  }

  const extension = uploading ? uploading.file.name.split('.').pop()?.toLowerCase() : null
  const typeLabel = extension ? ACCEPTED_LABELS[`application/${extension === 'pdf' ? 'pdf' : 'vnd.openxmlformats-officedocument.wordprocessingml.document'}`] ?? extension?.toUpperCase() : null

  return (
    <div className="w-full">
      {label && <label className="mb-2 block text-sm font-medium text-foreground">{label}</label>}

      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={() => setDragOver(false)}
        onClick={() => !disabled && inputRef.current?.click()}
        className={`relative cursor-pointer rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors
          ${dragOver ? 'border-accent bg-accent/5' : 'border-muted-foreground/25 hover:border-muted-foreground/50'}
          ${disabled ? 'cursor-not-allowed opacity-50' : ''}
        `}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(',')}
          onChange={onInputChange}
          className="hidden"
          disabled={disabled}
        />

        {uploading ? (
          <div className="space-y-2">
            <p className="text-sm font-medium text-foreground">{uploading.file.name}</p>
            <p className="text-xs text-muted-foreground">
              {typeLabel} &middot; {(uploading.file.size / 1024).toFixed(0)} KB
            </p>
            <div className="mx-auto h-2 w-48 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-accent transition-all duration-150"
                style={{ width: `${uploading.progress * 100}%` }}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              {uploading.progress >= 1 ? 'Hoàn tất' : `${Math.round(uploading.progress * 100)}%`}
            </p>
          </div>
        ) : (
          <>
            <svg
              className="mx-auto h-10 w-10 text-muted-foreground/50"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
            <p className="mt-3 text-sm font-medium text-foreground">
              Kéo thả file vào đây hoặc click để chọn
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              PDF, DOCX &middot; Tối đa 50MB
            </p>
          </>
        )}
      </div>

      {error && <p className="mt-2 text-xs text-destructive">{error}</p>}
    </div>
  )
}
