import { AskAgent } from "@/components/ask-agent"

export default function Page() {
  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-6 py-12 md:py-16">
        <header className="flex flex-col gap-2">
          <h1
            className="text-balance text-3xl font-bold tracking-tight md:text-4xl"
            style={{ color: "#F5532D" }}
          >
            AI Agent Bitovi&apos;s Blog
          </h1>
          <p className="text-pretty leading-relaxed text-muted-foreground">
            Ask a question and get an answer grounded in Bitovi&apos;s blog,
            complete with the references it used.
          </p>
        </header>

        <AskAgent />
      </div>
    </main>
  )
}
