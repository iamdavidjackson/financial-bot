import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import type { TradeSignal } from "../types"

const SIGNAL_STYLES: Record<TradeSignal, string> = {
  buy: "bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300",
  hold: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  sell: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
}

export function SignalBadge({
  signal,
  className,
}: {
  signal: TradeSignal
  className?: string
}) {
  return (
    <Badge className={cn(SIGNAL_STYLES[signal], className)}>{signal.toUpperCase()}</Badge>
  )
}
