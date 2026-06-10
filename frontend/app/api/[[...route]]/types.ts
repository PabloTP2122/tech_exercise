export type Source = { title: string; url: string }

export type AskResponse = {
  answer: string
  query_type: string
  sources: Source[]
}
