import { Hono } from "hono"
import { handle } from "hono/vercel"
import {
  MOCK_ENABLED,
  MOCK_DELAY_MS,
  getMockAnswer,
  getMockPreview,
} from "./mocks"
import type { AskResponse } from "./types"

export const runtime = "nodejs"

// Base URL of the FastAPI backend. Configurable via env, defaults to local :8000.
const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000"

const app = new Hono().basePath("/api")

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

/**
 * POST /api/ask
 * Proxies the question to the FastAPI + LangGraph backend at :8000/ask.
 */
app.post("/ask", async (c) => {
  let body: { question?: string }
  try {
    body = await c.req.json()
  } catch {
    return c.json({ error: "Invalid JSON body" }, 400)
  }

  const question = body?.question?.trim()
  if (!question) {
    return c.json({ error: "Field 'question' is required" }, 400)
  }

  // Mock mode: return canned JSON after a short delay so the UI's
  // isLoading state is exercised. Enabled via MOCK_API env var.
  if (MOCK_ENABLED) {
    await sleep(MOCK_DELAY_MS)
    return c.json(getMockAnswer(question))
  }

  try {
    const upstream = await fetch(`${BACKEND_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    })

    if (!upstream.ok) {
      const detail = await upstream.text().catch(() => "")
      return c.json(
        { error: `Backend responded with ${upstream.status}`, detail },
        502,
      )
    }

    const data = (await upstream.json()) as AskResponse
    return c.json(data)
  } catch (err) {
    return c.json(
      {
        error: "Failed to reach the RAG backend",
        detail: err instanceof Error ? err.message : String(err),
      },
      502,
    )
  }
})

/**
 * GET /api/preview?url=...
 * Server-side fetch of a linked page's OpenGraph / meta data for hover previews.
 */
app.get("/preview", async (c) => {
  const target = c.req.query("url")
  if (!target) {
    return c.json({ error: "Query param 'url' is required" }, 400)
  }

  let parsed: URL
  try {
    parsed = new URL(target)
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error("Unsupported protocol")
    }
  } catch {
    return c.json({ error: "Invalid url" }, 400)
  }

  // Mock mode: return stubbed OpenGraph data so hover previews work offline.
  if (MOCK_ENABLED) {
    await sleep(300)
    return c.json(getMockPreview(parsed))
  }

  try {
    const res = await fetch(parsed.toString(), {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (compatible; BitoviBlogBot/1.0; +https://www.bitovi.com)",
        Accept: "text/html",
      },
      signal: AbortSignal.timeout(6000),
    })

    if (!res.ok) {
      return c.json({ error: `Upstream ${res.status}` }, 200)
    }

    const html = await res.text()
    const preview = extractMeta(html, parsed)
    return c.json(preview)
  } catch {
    return c.json({ error: "preview_unavailable" }, 200)
  }
})

function extractMeta(html: string, base: URL) {
  const head = html.slice(0, 100_000)

  const meta = (prop: string) => {
    const patterns = [
      new RegExp(
        `<meta[^>]+(?:property|name)=["']${prop}["'][^>]*content=["']([^"']*)["']`,
        "i",
      ),
      new RegExp(
        `<meta[^>]+content=["']([^"']*)["'][^>]*(?:property|name)=["']${prop}["']`,
        "i",
      ),
    ]
    for (const re of patterns) {
      const m = head.match(re)
      if (m?.[1]) return decodeEntities(m[1].trim())
    }
    return undefined
  }

  const titleTag = head.match(/<title[^>]*>([^<]*)<\/title>/i)?.[1]

  const title =
    meta("og:title") ?? meta("twitter:title") ?? (titleTag ? decodeEntities(titleTag.trim()) : undefined)
  const description =
    meta("og:description") ?? meta("twitter:description") ?? meta("description")
  let image = meta("og:image") ?? meta("twitter:image")
  const siteName = meta("og:site_name")

  if (image && !/^https?:\/\//i.test(image)) {
    try {
      image = new URL(image, base).toString()
    } catch {
      image = undefined
    }
  }

  return {
    title,
    description,
    image,
    siteName,
    host: base.host,
  }
}

function decodeEntities(str: string) {
  return str
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x27;/g, "'")
    .replace(/&nbsp;/g, " ")
}

export const GET = handle(app)
export const POST = handle(app)
