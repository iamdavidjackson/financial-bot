import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import type { TradeConfirmationWidget } from "../types"

export function TradeConfirmationCard({
  data,
}: {
  data: TradeConfirmationWidget["data"]
}) {
  const isBuy = data.action === "buy"

  return (
    <Card size="sm" className="w-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-sm">Trade recorded</CardTitle>
        <Badge
          className={cn(
            isBuy
              ? "bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300"
              : "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300"
          )}
        >
          {data.action.toUpperCase()}
        </Badge>
      </CardHeader>
      <CardContent className="flex flex-col gap-1.5 text-sm">
        <div className="tabular-nums">
          {isBuy ? "Bought" : "Sold"} {data.shares} {data.ticker} @ ${data.price.toFixed(2)}
        </div>
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Cash balance</span>
          <span className="font-semibold tabular-nums">
            ${data.cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>
      </CardContent>
    </Card>
  )
}
