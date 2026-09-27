import { Chat } from "@/components/chat/chat";
import { QuoteCards } from "@/components/quotes/quote-cards";

export default function Home() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 bg-zinc-50 p-6 font-sans dark:bg-black">
      <QuoteCards />
      <Chat />
    </div>
  );
}
