export type ChatRole = "user" | "assistant"

export type TradeSignal = "buy" | "hold" | "sell"

export interface StockPredictionWidget {
  type: "stock_prediction"
  data: {
    ticker: string
    as_of_date: string
    current_close: number
    predicted_close_5d: number
    predicted_return_5d: number
    predicted_return_5d_pct: string
    signal: TradeSignal
  }
}

export interface TradeRecommendation {
  action: TradeSignal
  current_price: number
  current_holding_shares: number
  signal_percentile: number
  recommended_shares: number
  estimated_trade_value: number
}

export interface TradeRecommendationsWidget {
  type: "trade_recommendations"
  data: {
    as_of_date: string
    cash: number
    recommendations: Record<string, TradeRecommendation>
  }
}

export interface PortfolioWidget {
  type: "portfolio"
  data: {
    cash: number
    holdings: Record<string, number>
  }
}

export interface TradeConfirmationWidget {
  type: "trade_confirmation"
  data: {
    ticker: string
    action: "buy" | "sell"
    shares: number
    price: number
    cash: number
    holdings: Record<string, number>
  }
}

export type ChatWidget =
  | StockPredictionWidget
  | TradeRecommendationsWidget
  | PortfolioWidget
  | TradeConfirmationWidget

export interface ChatMessageType {
  id: string
  role: ChatRole
  content: string
  createdAt: Date
  widgets?: ChatWidget[]
}
