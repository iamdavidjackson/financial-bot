import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { PortfolioWidget } from "../types"

export function PortfolioCard({ data }: { data: PortfolioWidget["data"] }) {
  const holdings = Object.entries(data.holdings)

  return (
    <Card size="sm" className="w-full">
      <CardHeader>
        <CardTitle className="text-sm">Portfolio</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Cash</span>
          <span className="font-semibold tabular-nums">
            ${data.cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>
        {holdings.length > 0 ? (
          <ul className="flex flex-col divide-y divide-border">
            {holdings.map(([ticker, shares]) => (
              <li key={ticker} className="flex items-center justify-between py-1.5 text-sm">
                <span className="font-medium">{ticker}</span>
                <span className="tabular-nums">{shares} shares</span>
              </li>
            ))}
          </ul>
        ) : (
          <div className="text-xs text-muted-foreground">No open positions.</div>
        )}
      </CardContent>
    </Card>
  )
}
