import { ArrowDownRight, ArrowUpRight } from "lucide-react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { StockPredictionWidget } from "../types"
import { SignalBadge } from "./signal-badge"

export function StockPredictionCard({ data }: { data: StockPredictionWidget["data"] }) {
  const up = data.predicted_return_5d >= 0
  const Arrow = up ? ArrowUpRight : ArrowDownRight
  const changeColor = up ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"

  return (
    <Card size="sm" className="w-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-sm">{data.ticker} · 5-day outlook</CardTitle>
        <SignalBadge signal={data.signal} />
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        <div className="flex items-end gap-3">
          <div>
            <div className="text-xs text-muted-foreground">Current close</div>
            <div className="text-lg font-semibold tabular-nums">
              ${data.current_close.toFixed(2)}
            </div>
          </div>
          <Arrow className={`mb-1 size-5 ${changeColor}`} />
          <div>
            <div className="text-xs text-muted-foreground">Predicted close</div>
            <div className="text-lg font-semibold tabular-nums">
              ${data.predicted_close_5d.toFixed(2)}
            </div>
          </div>
          <div className={`ml-auto text-sm font-semibold tabular-nums ${changeColor}`}>
            {data.predicted_return_5d_pct}
          </div>
        </div>
        <div className="text-xs text-muted-foreground">
          Model prediction as of {data.as_of_date} · not financial advice
        </div>
      </CardContent>
    </Card>
  )
}
