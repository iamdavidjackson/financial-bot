"use client"

import { useState } from "react"
import { toast } from "sonner"

import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { ChatInput } from "./chat-input"
import { ChatMessageList } from "./chat-message-list"
import type { ChatMessageType } from "./types"

const INITIAL_MESSAGES: ChatMessageType[] = [
  {
    id: "1",
    role: "assistant",
    content: "Hi! How can I help you today?",
    createdAt: new Date(),
  },
]

async function fetchAssistantReply(history: ChatMessageType[]): Promise<string> {
  // Placeholder — replace with a call to your chat API endpoint.
  await new Promise((resolve) => setTimeout(resolve, 800))
  return `You said: "${history[history.length - 1]?.content}"`
}

export function Chat() {
  const [messages, setMessages] = useState<ChatMessageType[]>(INITIAL_MESSAGES)
  const [isLoading, setIsLoading] = useState(false)

  async function sendToAssistant(history: ChatMessageType[]) {
    setIsLoading(true)
    try {
      const reply = await fetchAssistantReply(history)
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: reply,
          createdAt: new Date(),
        },
      ])
    } catch {
      toast.error("Something went wrong. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  function handleSend(content: string) {
    const userMessage: ChatMessageType = {
      id: crypto.randomUUID(),
      role: "user",
      content,
      createdAt: new Date(),
    }
    const nextMessages = [...messages, userMessage]
    setMessages(nextMessages)
    void sendToAssistant(nextMessages)
  }

  return (
    <Card className="flex h-[32rem] w-full max-w-2xl flex-col gap-0 overflow-hidden py-0">
      <CardHeader className="py-3">
        <CardTitle>Chat</CardTitle>
        <CardAction>
          <Badge variant="secondary">{isLoading ? "Thinking…" : "Online"}</Badge>
        </CardAction>
      </CardHeader>
      <Separator />
      <ChatMessageList messages={messages} isLoading={isLoading} />
      <CardFooter>
        <ChatInput onSend={handleSend} disabled={isLoading} />
      </CardFooter>
    </Card>
  )
}
