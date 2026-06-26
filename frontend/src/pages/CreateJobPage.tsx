import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { DocumentUploader } from '@/components/upload/DocumentUploader'
import { uploadDocument } from '@/api/documents'
import { createJob } from '@/api/jobs'
import { getLLMSettings } from '@/api/settings'
import type { DocumentMeta } from '@/types/document'

export function CreateJobPage() {
  const navigate = useNavigate()
  const [docV1, setDocV1] = useState<DocumentMeta | null>(null)
  const [docV2, setDocV2] = useState<DocumentMeta | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const { data: settings } = useQuery({
    queryKey: ['llm-settings'],
    queryFn: getLLMSettings,
  })

  const jobMutation = useMutation({
    mutationFn: createJob,
    onSuccess: (data) => {
      navigate(`/report/${data.id}`)
    },
    onError: () => {
      setUploadError('Không thể tạo phiên so sánh. Vui lòng thử lại.')
    },
  })

  const handleSubmit = () => {
    if (!docV1 || !docV2) {
      setUploadError('Vui lòng tải lên cả hai tài liệu')
      return
    }
    setUploadError(null)
    jobMutation.mutate({
      document_v1_id: docV1.id,
      document_v2_id: docV2.id,
    })
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <h1 className="font-serif text-2xl font-semibold">So sánh mới</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Tải lên hai phiên bản văn bản pháp luật để phân tích sự khác biệt
      </p>

      <div className="mt-8 space-y-6">
        <div className="grid gap-6 md:grid-cols-2">
          <div className="rounded-lg border bg-white p-6">
            <h2 className="font-serif text-lg font-medium">Tài liệu gốc (V1)</h2>
            <p className="mb-4 text-xs text-muted-foreground">Văn bản pháp luật phiên bản cũ</p>
            <DocumentUploader
              onUpload={async (file) => {
                setUploadError(null)
                const doc = await uploadDocument(file)
                setDocV1(doc)
              }}
            />
            {docV1 && (
              <p className="mt-3 text-sm text-green-700">
                Đã tải lên: {docV1.original_filename}
              </p>
            )}
          </div>

          <div className="rounded-lg border bg-white p-6">
            <h2 className="font-serif text-lg font-medium">Tài liệu sửa đổi (V2)</h2>
            <p className="mb-4 text-xs text-muted-foreground">Văn bản pháp luật phiên bản mới</p>
            <DocumentUploader
              onUpload={async (file) => {
                setUploadError(null)
                const doc = await uploadDocument(file)
                setDocV2(doc)
              }}
            />
            {docV2 && (
              <p className="mt-3 text-sm text-green-700">
                Đã tải lên: {docV2.original_filename}
              </p>
            )}
          </div>
        </div>

        {uploadError && (
          <div className="rounded-md bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {uploadError}
          </div>
        )}

        <button
          onClick={handleSubmit}
          disabled={jobMutation.isPending || !docV1 || !docV2}
          className="rounded-md bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {jobMutation.isPending ? 'Đang tạo...' : 'Bắt đầu so sánh'}
        </button>
      </div>
    </div>
  )
}
