import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation } from '@tanstack/react-query'
import { DocumentUploader } from '@/components/upload/DocumentUploader'
import { uploadDocument } from '@/api/documents'
import { createJob } from '@/api/jobs'
import type { DocumentMeta } from '@/types/document'

const jobSchema = z.object({
  name: z.string().min(1, 'Vui lòng nhập tên phiên so sánh'),
})

type JobForm = z.infer<typeof jobSchema>

export function CreateJobPage() {
  const navigate = useNavigate()
  const [docV1, setDocV1] = useState<DocumentMeta | null>(null)
  const [docV2, setDocV2] = useState<DocumentMeta | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<JobForm>({
    resolver: zodResolver(jobSchema),
    defaultValues: { name: `So sánh ${new Date().toLocaleDateString('vi-VN')}` },
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

  const onSubmit = (data: JobForm) => {
    if (!docV1 || !docV2) {
      setUploadError('Vui lòng tải lên cả hai tài liệu')
      return
    }
    setUploadError(null)
    jobMutation.mutate({
      name: data.name,
      doc_v1_id: docV1.id,
      doc_v2_id: docV2.id,
    })
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <h1 className="font-serif text-2xl font-semibold">So sánh mới</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Tải lên hai phiên bản văn bản pháp luật để phân tích sự khác biệt
      </p>

      <form onSubmit={handleSubmit(onSubmit)} className="mt-8 space-y-6">
        <div className="grid gap-6 md:grid-cols-2">
          <div className="rounded-lg border bg-white p-6">
            <h2 className="font-serif text-lg font-medium">Tài liệu gốc (V1)</h2>
            <p className="mb-4 text-xs text-muted-foreground">Văn bản pháp luật phiên bản cũ</p>
            <DocumentUploader
              label=""
              onUpload={async (file) => {
                setUploadError(null)
                const doc = await uploadDocument(file)
                setDocV1(doc)
              }}
            />
            {docV1 && (
              <p className="mt-3 text-sm text-green-700">
                Đã tải lên: {docV1.filename}
              </p>
            )}
          </div>

          <div className="rounded-lg border bg-white p-6">
            <h2 className="font-serif text-lg font-medium">Tài liệu sửa đổi (V2)</h2>
            <p className="mb-4 text-xs text-muted-foreground">Văn bản pháp luật phiên bản mới</p>
            <DocumentUploader
              label=""
              onUpload={async (file) => {
                setUploadError(null)
                const doc = await uploadDocument(file)
                setDocV2(doc)
              }}
            />
            {docV2 && (
              <p className="mt-3 text-sm text-green-700">
                Đã tải lên: {docV2.filename}
              </p>
            )}
          </div>
        </div>

        <div>
          <label htmlFor="jobName" className="block text-sm font-medium text-foreground">
            Tên phiên so sánh
          </label>
          <input
            id="jobName"
            type="text"
            className="mt-1 block w-full max-w-md rounded-md border px-3 py-2 text-sm focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
            {...register('name')}
          />
          {errors.name && (
            <p className="mt-1 text-xs text-destructive">{errors.name.message}</p>
          )}
        </div>

        {uploadError && (
          <div className="rounded-md bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {uploadError}
          </div>
        )}

        <button
          type="submit"
          disabled={jobMutation.isPending || !docV1 || !docV2}
          className="rounded-md bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {jobMutation.isPending ? 'Đang tạo...' : 'Bắt đầu so sánh'}
        </button>
      </form>
    </div>
  )
}
