import { Chat } from "@/components/chat/chat";

export default function Home() {
  return (
    <div className="flex flex-1 items-center justify-center bg-zinc-50 p-6 font-sans dark:bg-black">
      <Chat />
    </div>
  );
}
