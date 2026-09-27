"use client"

import { useEffect, useState } from "react"
import { ArrowDownRight, ArrowUpRight } from "lucide-react"

import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"

type Quote = {
  ticker: string
  name: string
  price: number
  previous_close: number
  change: number
  change_percent: number
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
const PLACEHOLDER_COUNT = 3

export function QuoteCards() {
  const [quotes, setQuotes] = useState<Quote[] | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    fetch(`${API_URL}/quotes`)
      .then((response) => {
        if (!response.ok) throw new Error(`Quotes request failed with status ${response.status}`)
        return response.json()
      })
      .then(setQuotes)
      .catch(() => setFailed(true))
  }, [])

  if (failed) return null

  const skeletons = Array.from({ length: PLACEHOLDER_COUNT }, (_, i) => (
    <Skeleton key={i} className="h-[4.5rem] rounded-xl" />
  ));

  const quotesMap = (quote: Quote) => <QuoteCard key={quote.ticker} quote={quote} />;
  return (
    <div className="grid w-full max-w-2xl grid-cols-1 gap-3 sm:grid-cols-3">
      {quotes ? quotes.map(quotesMap) : skeletons}
    </div>
  )
}

function QuoteCard({ quote }: { quote: Quote }) {
  const up = quote.change >= 0
  const Arrow = up ? ArrowUpRight : ArrowDownRight
  const changeColor = up ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"

  return (
    <Card size="sm">
      <CardContent className="flex flex-col gap-1">
        <div className="flex items-baseline justify-between gap-2">
          <span className="font-semibold">{quote.ticker}</span>
          <span className="truncate text-xs text-muted-foreground">{quote.name}</span>
        </div>
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-lg font-semibold tabular-nums">${quote.price.toFixed(2)}</span>
          <span className={`flex items-center text-xs font-medium tabular-nums ${changeColor}`}>
            <Arrow className="size-3.5" />
            {up ? "+" : ""}
            {quote.change.toFixed(2)} ({up ? "+" : ""}
            {quote.change_percent.toFixed(2)}%)
          </span>
        </div>
      </CardContent>
    </Card>
  )
}
