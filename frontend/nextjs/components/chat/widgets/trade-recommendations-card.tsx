import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { TradeRecommendationsWidget } from "../types"
import { SignalBadge } from "./signal-badge"

export function TradeRecommendationsCard({
  data,
}: {
  data: TradeRecommendationsWidget["data"]
}) {
  const rows = Object.entries(data.recommendations)

  return (
    <Card size="sm" className="w-full">
      <CardHeader>
        <CardTitle className="text-sm">Trade recommendations</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>As of {data.as_of_date}</span>
          <span className="tabular-nums">
            Cash ${data.cash.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
        </div>
        <ul className="flex flex-col divide-y divide-border">
          {rows.map(([ticker, rec]) => {
            const sizeable = rec.action !== "hold" && rec.recommended_shares > 0
            return (
              <li key={ticker} className="flex items-center justify-between gap-2 py-1.5">
                <div className="min-w-0">
                  <div className="text-sm font-medium">{ticker}</div>
                  <div className="text-xs text-muted-foreground tabular-nums">
                    ${rec.current_price.toFixed(2)} · {rec.current_holding_shares} sh · signal{" "}
                    {(rec.signal_percentile * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="flex flex-col items-end gap-0.5">
                  <SignalBadge signal={rec.action} />
                  {sizeable && (
                    <div className="text-xs text-muted-foreground tabular-nums">
                      ~{rec.recommended_shares} sh (~$
                      {rec.estimated_trade_value.toLocaleString(undefined, {
                        maximumFractionDigits: 0,
                      })}
                      )
                    </div>
                  )}
                </div>
              </li>
            )
          })}
        </ul>
        <div className="text-xs text-muted-foreground">
          Trained model suggestion · not financial advice
        </div>
      </CardContent>
    </Card>
  )
}
