import axios from "axios"

export type Source = {
  title: string
  url: string
}

export type AskResponse = {
  answer: string
  query_type: string
  sources: Source[]
}

export type PreviewData = {
  title?: string
  description?: string
  image?: string
  siteName?: string
  host?: string
  error?: string
}

const http = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
})

// Mutation fetcher for SWR's useSWRMutation (POST /api/ask).
export async function askFetcher(
  url: string,
  { arg }: { arg: { question: string } },
): Promise<AskResponse> {
  const { data } = await http.post<AskResponse>(url, arg)
  return data
}

// GET fetcher used by SWR for hover previews (GET /api/preview?url=...).
export async function previewFetcher(url: string): Promise<PreviewData> {
  const { data } = await http.get<PreviewData>(url)
  return data
}
