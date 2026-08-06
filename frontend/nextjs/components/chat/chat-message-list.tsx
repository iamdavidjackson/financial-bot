"use client"

import { useEffect, useRef } from "react"

import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Skeleton } from "@/components/ui/skeleton"
import { ChatMessage } from "./chat-message"
import type { ChatMessageType } from "./types"

interface ChatMessageListProps {
  messages: ChatMessageType[]
  isLoading: boolean
}

export function ChatMessageList({ messages, isLoading }: ChatMessageListProps) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isLoading])

  return (
    <ScrollArea className="flex-1">
      <div className="flex flex-col gap-4 px-4 py-4">
        {messages.map((message) => (
          <ChatMessage key={message.id} message={message} />
        ))}
        {isLoading && (
          <div className="flex items-start gap-2.5">
            <Avatar size="sm">
              <AvatarFallback>AI</AvatarFallback>
            </Avatar>
            <div className="flex flex-col gap-1.5 rounded-2xl bg-muted px-3.5 py-2.5">
              <Skeleton className="h-3 w-32" />
              <Skeleton className="h-3 w-20" />
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>
    </ScrollArea>
  )
}
