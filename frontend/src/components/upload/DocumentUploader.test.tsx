import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { DocumentUploader } from './DocumentUploader'

function uploadFile(file: File) {
  Object.defineProperty(HTMLInputElement.prototype, 'files', {
    get() { return [file] },
    configurable: true,
  })
  const input = document.querySelector('input[type="file"]') as HTMLInputElement
  fireEvent.change(input)
}

describe('DocumentUploader', () => {
  it('renders the drop zone', () => {
    render(<DocumentUploader onUpload={vi.fn()} />)
    expect(screen.getByText(/Kéo thả file vào đây/i)).toBeInTheDocument()
    expect(screen.getByText(/PDF, DOCX/i)).toBeInTheDocument()
  })

  it('shows label when provided', () => {
    render(<DocumentUploader onUpload={vi.fn()} label="Tài liệu V1" />)
    expect(screen.getByText('Tài liệu V1')).toBeInTheDocument()
  })

  it('shows error for non-PDF/DOCX files', async () => {
    render(<DocumentUploader onUpload={vi.fn()} />)

    const file = new File(['content'], 'test.txt', { type: 'text/plain' })
    uploadFile(file)

    await waitFor(() => {
      expect(screen.getByText(/Chỉ chấp nhận file PDF hoặc DOCX/i)).toBeInTheDocument()
    })
  })

  it('shows error for files over 50MB', async () => {
    render(<DocumentUploader onUpload={vi.fn()} />)

    const largeContent = new Uint8Array(51 * 1024 * 1024)
    const file = new File([largeContent], 'large.pdf', { type: 'application/pdf' })
    uploadFile(file)

    await waitFor(() => {
      expect(screen.getByText(/không được vượt quá 50MB/i)).toBeInTheDocument()
    })
  })

  it('calls onUpload for valid PDF', async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined)
    render(<DocumentUploader onUpload={onUpload} />)

    const file = new File(['pdf content'], 'test.pdf', { type: 'application/pdf' })
    uploadFile(file)

    await waitFor(() => {
      expect(onUpload).toHaveBeenCalledTimes(1)
    })
  })

  it('shows progress bar during upload', async () => {
    const onUpload = vi.fn(() => new Promise<void>(() => {}))
    render(<DocumentUploader onUpload={onUpload} />)

    const file = new File(['pdf content'], 'test.pdf', { type: 'application/pdf' })
    uploadFile(file)

    await waitFor(() => {
      expect(screen.getByText(/test\.pdf/)).toBeInTheDocument()
    })
  })
})
