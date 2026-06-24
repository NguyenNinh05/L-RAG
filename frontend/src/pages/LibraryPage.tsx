import { Link } from 'react-router-dom'

export function LibraryPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <div className="flex items-center justify-between">
        <h1 className="font-serif text-2xl font-semibold">Thư viện tài liệu</h1>
        <Link
          to="/compare"
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
        >
          So sánh mới
        </Link>
      </div>

      <div className="mt-8 py-16 text-center">
        <p className="text-sm text-muted-foreground">
          Tài liệu đã tải lên sẽ xuất hiện ở đây sau khi backend tích hợp.
        </p>
        <Link to="/compare" className="mt-2 inline-block text-sm font-medium text-accent hover:underline">
          Bắt đầu so sánh đầu tiên
        </Link>
      </div>
    </div>
  )
}
