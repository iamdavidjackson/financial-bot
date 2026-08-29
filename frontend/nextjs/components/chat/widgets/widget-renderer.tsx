import type { ChatWidget } from "../types"
import { PortfolioCard } from "./portfolio-card"
import { StockPredictionCard } from "./stock-prediction-card"
import { TradeConfirmationCard } from "./trade-confirmation-card"
import { TradeRecommendationsCard } from "./trade-recommendations-card"

export function WidgetRenderer({ widget }: { widget: ChatWidget }) {
  switch (widget.type) {
    case "stock_prediction":
      return <StockPredictionCard data={widget.data} />
    case "trade_recommendations":
      return <TradeRecommendationsCard data={widget.data} />
    case "portfolio":
      return <PortfolioCard data={widget.data} />
    case "trade_confirmation":
      return <TradeConfirmationCard data={widget.data} />
    default:
      return null
  }
}
