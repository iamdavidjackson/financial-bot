export function SiteFooter() {
  const year = new Date().getFullYear()
  return (
    <footer className="border-t bg-background">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-1 px-6 py-4 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
        <p>&copy; {year} Jackson Investments. All rights reserved.</p>
        <p>For informational purposes only. Not financial advice.</p>
      </div>
    </footer>
  )
}
