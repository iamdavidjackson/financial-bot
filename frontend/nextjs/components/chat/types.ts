export type ChatRole = "user" | "assistant"

export interface ChatMessageType {
  id: string
  role: ChatRole
  content: string
  createdAt: Date
}
