"use client"

import { useState } from "react"
import useSWR from "swr"
import { ExternalLink, Globe } from "lucide-react"
import { previewFetcher, type Source } from "@/lib/api"

export function SourceLink({ source, index }: { source: Source; index: number }) {
  const [open, setOpen] = useState(false)

  // Only fetch the preview once the user hovers/focuses the link.
  const { data, isLoading } = useSWR(
    open ? `/preview?url=${encodeURIComponent(source.url)}` : null,
    previewFetcher,
    { revalidateOnFocus: false, dedupingInterval: 60_000 },
  )

  const hasPreview = data && !data.error && (data.title || data.description || data.image)

  return (
    <div
      className="relative"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <a
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="group flex items-start gap-3 rounded-lg border border-border bg-card px-4 py-3 transition-colors hover:border-brand-bright hover:bg-secondary focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <span className="mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-md bg-brand-teal text-xs font-semibold text-white">
          {index + 1}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate font-medium text-brand-deep">
            {source.title}
          </span>
          <span className="mt-0.5 block truncate text-sm text-muted-foreground">
            {source.url}
          </span>
        </span>
        <ExternalLink className="mt-1 size-4 shrink-0 text-brand-teal opacity-0 transition-opacity group-hover:opacity-100" />
      </a>

      {open && (
        <div
          role="tooltip"
          className="absolute left-0 top-full z-20 mt-2 w-80 max-w-[calc(100vw-3rem)] overflow-hidden rounded-xl border border-border bg-popover shadow-lg"
        >
          {isLoading && (
            <div className="flex items-center gap-2 px-4 py-5 text-sm text-muted-foreground">
              <Globe className="size-4 animate-pulse text-brand-teal" />
              Loading preview…
            </div>
          )}

          {!isLoading && hasPreview && (
            <div>
              {data?.image && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={data.image}
                  alt=""
                  className="h-36 w-full object-cover"
                  crossOrigin="anonymous"
                />
              )}
              <div className="space-y-1 p-4">
                {data?.siteName && (
                  <span className="text-xs font-medium uppercase tracking-wide text-brand-bright">
                    {data.siteName}
                  </span>
                )}
                {data?.title && (
                  <p className="text-pretty font-semibold leading-snug text-brand-deep">
                    {data.title}
                  </p>
                )}
                {data?.description && (
                  <p className="line-clamp-3 text-pretty text-sm leading-relaxed text-muted-foreground">
                    {data.description}
                  </p>
                )}
                <span className="block truncate pt-1 text-xs text-muted-foreground">
                  {data?.host ?? source.url}
                </span>
              </div>
            </div>
          )}

          {!isLoading && !hasPreview && (
            <div className="flex items-center gap-2 px-4 py-5 text-sm text-muted-foreground">
              <Globe className="size-4 text-brand-teal" />
              Preview unavailable — click to open the source.
            </div>
          )}
        </div>
      )}
    </div>
  )
}
