"use client"

import { cn } from "@/lib/utils"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import type { ChatMessageType } from "./types"
import { WidgetRenderer } from "./widgets/widget-renderer"

interface ChatMessageProps {
  message: ChatMessageType
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user"
  const hasWidgets = !isUser && (message.widgets?.length ?? 0) > 0

  return (
    <div className={cn("flex items-start gap-2.5", isUser && "flex-row-reverse")}>
      <Avatar size="sm">
        <AvatarFallback>{isUser ? "You" : "AI"}</AvatarFallback>
      </Avatar>
      <div
        className={cn(
          "flex min-w-0 flex-col gap-2",
          isUser ? "max-w-[85%] items-end" : "flex-1"
        )}
      >
        {message.content && (
          <div
            className={cn(
              "w-fit rounded-2xl px-3.5 py-2 text-sm whitespace-pre-wrap",
              isUser ? "bg-primary text-primary-foreground" : "bg-muted text-foreground"
            )}
          >
            {message.content}
          </div>
        )}
        {hasWidgets &&
          message.widgets!.map((widget, index) => (
            <WidgetRenderer key={index} widget={widget} />
          ))}
      </div>
    </div>
  )
}
