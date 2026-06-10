"use client"

import { useState, type KeyboardEvent } from "react"
import useSWRMutation from "swr/mutation"
import { ArrowUp, Loader2, Sparkles, BookOpen, AlertCircle } from "lucide-react"
import { askFetcher, type AskResponse } from "@/lib/api"
import { SourceLink } from "@/components/source-link"

export function AskAgent() {
  const [question, setQuestion] = useState("")

  const { trigger, data, error, isMutating } = useSWRMutation<
    AskResponse,
    Error,
    string,
    { question: string }
  >("/ask", askFetcher)

  const submit = async () => {
    const q = question.trim()
    if (!q || isMutating) return
    try {
      await trigger({ question: q })
    } catch {
      // error is surfaced via the `error` value from useSWRMutation
    }
  }

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      void submit()
    }
  }

  return (
    <section className="flex w-full flex-col gap-6">
      {/* Input row — no <form>, submit on click + Enter */}
      <div className="flex items-center gap-2 rounded-2xl border border-border bg-card p-2 shadow-sm focus-within:border-brand-bright focus-within:ring-2 focus-within:ring-ring">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={isMutating}
          aria-label="Ask a question about the Bitovi blog"
          placeholder="Ask anything about the Bitovi blog…"
          className="min-w-0 flex-1 bg-transparent px-3 py-2.5 text-brand-deep placeholder:text-muted-foreground focus:outline-none disabled:opacity-60"
        />
        <button
          type="button"
          onClick={() => void submit()}
          disabled={isMutating || !question.trim()}
          aria-label="Submit question"
          className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-brand-teal text-white transition-colors hover:bg-brand-bright disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isMutating ? (
            <Loader2 className="size-5 animate-spin" />
          ) : (
            <ArrowUp className="size-5" />
          )}
        </button>
      </div>

      {/* Loading state */}
      {isMutating && (
        <div className="flex items-center gap-3 rounded-xl border border-border bg-muted px-4 py-3 text-brand-deep">
          <Loader2 className="size-4 animate-spin text-brand-teal" />
          <span className="text-sm">Searching the blog and composing an answer…</span>
        </div>
      )}

      {/* Error state */}
      {error && !isMutating && (
        <div className="flex items-start gap-3 rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-destructive">
          <AlertCircle className="mt-0.5 size-4 shrink-0" />
          <div className="text-sm">
            <p className="font-medium">Something went wrong.</p>
            <p className="opacity-80">
              Couldn&apos;t reach the agent. Make sure the backend is running on
              port 8000 and try again.
            </p>
          </div>
        </div>
      )}

      {/* Answer */}
      {data && !isMutating && (
        <div className="flex flex-col gap-6 rounded-2xl border border-border bg-card p-6 shadow-sm">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2 text-brand-deep">
              <Sparkles className="size-5 text-brand-teal" />
              <h2 className="text-lg font-semibold">Answer</h2>
            </div>
            {data.query_type && (
              <span className="inline-flex items-center rounded-full bg-brand-bright/15 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-brand-teal">
                {data.query_type}
              </span>
            )}
          </div>

          <p className="whitespace-pre-wrap text-pretty leading-relaxed text-brand-deep">
            {data.answer}
          </p>

          {data.sources?.length > 0 && (
            <div className="flex flex-col gap-3 border-t border-border pt-5">
              <div className="flex items-center gap-2 text-brand-deep">
                <BookOpen className="size-4 text-brand-teal" />
                <h3 className="text-sm font-semibold uppercase tracking-wide">
                  References
                </h3>
              </div>
              <div className="flex flex-col gap-2">
                {data.sources.map((source, i) => (
                  <SourceLink key={`${source.url}-${i}`} source={source} index={i} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
