import { Chat } from "@/components/chat/chat";
import { QuoteCards } from "@/components/quotes/quote-cards";

const EXAMPLE_QUESTIONS = [
  "How will NEE perform in 5 days?",
  "Tell me what trades I should make with my portfolio",
];

export default function Home() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 bg-zinc-50 p-6 font-sans dark:bg-black">
      <QuoteCards />
      <Chat />
      <p className="w-full max-w-2xl text-center text-xs text-muted-foreground">
        Try asking{" "}
        {EXAMPLE_QUESTIONS.map((question, index) => (
          <span key={question}>
            {index > 0 && " or "}
            <span className="font-medium text-foreground">&ldquo;{question}&rdquo;</span>
          </span>
        ))}
      </p>
    </div>
  );
}
