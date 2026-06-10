import type { AskResponse } from "./types"

/**
 * Toggle mock mode here. When true, the API returns simulated agent
 * responses so the UI runs fully offline, without the FastAPI +
 * LangGraph backend on :8000. Flip to false to proxy to the real backend.
 */
export const MOCK_ENABLED = false

// Simulated network latency (ms) so the UI's isLoading state is visible.
export const MOCK_DELAY_MS = 900

type Scenario = {
  /** Keywords (lowercase) that, if all present, select this scenario. */
  match: string[]
  build: (question: string) => AskResponse
}

/**
 * Keyword-matched mock scenarios, each tied to a distinct agent query
 * strategy. The first scenario whose keywords all appear in the question
 * wins; anything unmatched gets the generic fallback below.
 */
const SCENARIOS: Scenario[] = [
  // semantic QA: similarity search -> grounded answer
  {
    match: ["e2e", "testing"],
    build: () => ({
      answer:
        "For end-to-end testing, the Bitovi blog most often recommends Playwright and Cypress as the primary " +
        "browser-automation tools, with WebdriverIO called out for teams that need broad real-device coverage. " +
        "Playwright is favored for its speed, parallelism, and first-class TypeScript support, while Cypress is " +
        "highlighted for its developer experience and time-travel debugging. The posts pair these with CI " +
        "integration and the Page Object pattern to keep suites maintainable.",
      query_type: "semantic_qa",
      sources: [
        {
          title: "Playwright vs. Cypress: Choosing an E2E Testing Tool",
          url: "https://www.bitovi.com/blog/playwright-vs-cypress-choosing-an-e2e-testing-tool",
        },
        {
          title: "End-to-End Testing Best Practices",
          url: "https://www.bitovi.com/blog/end-to-end-testing-best-practices",
        },
      ],
    }),
  },
  // enumeration: metadata filter -> complete list
  {
    match: ["all articles", "devops"],
    build: () => ({
      answer:
        "I found 5 articles tagged DevOps in the Bitovi blog. Here is the complete list.",
      query_type: "enumeration",
      sources: [
        {
          title: "A Practical Guide to GitOps with ArgoCD",
          url: "https://www.bitovi.com/blog/a-practical-guide-to-gitops-with-argocd",
        },
        {
          title: "Kubernetes Deployment Strategies Explained",
          url: "https://www.bitovi.com/blog/kubernetes-deployment-strategies-explained",
        },
        {
          title: "Infrastructure as Code with Terraform",
          url: "https://www.bitovi.com/blog/infrastructure-as-code-with-terraform",
        },
        {
          title: "Building CI/CD Pipelines That Scale",
          url: "https://www.bitovi.com/blog/building-ci-cd-pipelines-that-scale",
        },
        {
          title: "Observability: Logs, Metrics, and Traces",
          url: "https://www.bitovi.com/blog/observability-logs-metrics-and-traces",
        },
      ],
    }),
  },
  // count: metadata count -> accurate number
  {
    match: ["how many", "ai"],
    build: () => ({
      answer:
        "There are 12 articles about AI in the Bitovi blog, spanning topics like LLM application architecture, " +
        "RAG pipelines, prompt engineering, and integrating AI into existing product workflows.",
      query_type: "count",
      sources: [
        {
          title: "Browse all AI articles on the Bitovi blog",
          url: "https://www.bitovi.com/blog/tags/ai",
        },
      ],
    }),
  },
  // recency: live RSS feed -> genuinely newest article
  {
    match: ["latest", "blog post"],
    build: () => ({
      answer:
        "The latest post on the Bitovi blog is \"Server Components in Practice: Lessons from Production.\" " +
        "Published this week, it walks through real-world patterns for adopting React Server Components, " +
        "the data-fetching trade-offs the team encountered, and how to incrementally migrate an existing app.",
      query_type: "recency",
      sources: [
        {
          title: "Server Components in Practice: Lessons from Production",
          url: "https://www.bitovi.com/blog/server-components-in-practice-lessons-from-production",
        },
      ],
    }),
  },
  // semantic QA: the original Angular Signals example
  {
    match: ["signal"],
    build: (question) => ({
      answer:
        `Based on Bitovi's blog, here's what I found regarding "${question}":\n\n` +
        "Angular Signals provide a reactive primitive for managing state with fine-grained change detection. " +
        "You create one with `signal(initialValue)`, read it by calling it as a function, and update it with " +
        "`.set()` or `.update()`. Derived state uses `computed()`, and side effects run inside `effect()`. " +
        "Signals integrate with the template so only the specific DOM bindings that depend on a changed signal " +
        "are re-evaluated, which improves performance over traditional zone-based change detection.",
      query_type: "semantic_qa",
      sources: [
        {
          title: "A Comprehensive Guide to Angular Signals",
          url: "https://www.bitovi.com/blog/a-comprehensive-guide-to-angular-signals",
        },
        {
          title: "Angular Change Detection: How It Works",
          url: "https://www.bitovi.com/blog/angular-change-detection-how-it-works",
        },
        {
          title: "Managing State in Modern Angular Applications",
          url: "https://www.bitovi.com/blog/managing-state-in-modern-angular-applications",
        },
      ],
    }),
  },
]

/**
 * Returns a simulated agent response. Matches the question against the
 * keyword scenarios above; if nothing matches, returns a single generic
 * fallback answer with no sources.
 */
export function getMockAnswer(question: string): AskResponse {
  const q = question.toLowerCase()
  const scenario = SCENARIOS.find((s) => s.match.every((kw) => q.includes(kw)))
  if (scenario) {
    return scenario.build(question)
  }

  return {
    answer:
      "I couldn't find anything in the Bitovi blog that directly answers that. Try rephrasing your question " +
      "or asking about Angular, React, DevOps, AI, or testing topics the blog covers.",
    query_type: "no_answer",
    sources: [],
  }
}

type PreviewData = {
  title?: string
  description?: string
  image?: string
  siteName?: string
  host: string
}

/**
 * Stubbed OpenGraph-style metadata keyed by url, so hover previews work
 * offline. Falls back to a generic Bitovi card for unknown urls.
 */
const MOCK_PREVIEWS: Record<string, Omit<PreviewData, "host">> = {
  "https://www.bitovi.com/blog/a-comprehensive-guide-to-angular-signals": {
    title: "A Comprehensive Guide to Angular Signals",
    description:
      "Learn how Angular Signals work, when to use signal(), computed(), and effect(), and how they improve change detection performance.",
    siteName: "Bitovi",
  },
  "https://www.bitovi.com/blog/angular-change-detection-how-it-works": {
    title: "Angular Change Detection: How It Works",
    description:
      "A deep dive into Angular's change detection mechanism, zones, and how Signals offer a more granular alternative.",
    siteName: "Bitovi",
  },
  "https://www.bitovi.com/blog/managing-state-in-modern-angular-applications": {
    title: "Managing State in Modern Angular Applications",
    description:
      "Patterns and trade-offs for state management in Angular, from services to Signals to NgRx.",
    siteName: "Bitovi",
  },
  // semantic QA — E2E testing
  "https://www.bitovi.com/blog/playwright-vs-cypress-choosing-an-e2e-testing-tool": {
    title: "Playwright vs. Cypress: Choosing an E2E Testing Tool",
    description:
      "A head-to-head comparison of Playwright and Cypress covering speed, debugging, parallelism, and CI integration.",
    siteName: "Bitovi",
  },
  "https://www.bitovi.com/blog/end-to-end-testing-best-practices": {
    title: "End-to-End Testing Best Practices",
    description:
      "How to keep E2E suites fast and maintainable with the Page Object pattern, stable selectors, and smart CI setup.",
    siteName: "Bitovi",
  },
  // enumeration — DevOps articles
  "https://www.bitovi.com/blog/a-practical-guide-to-gitops-with-argocd": {
    title: "A Practical Guide to GitOps with ArgoCD",
    description:
      "Declarative, Git-driven Kubernetes deployments using ArgoCD, with sync strategies and rollback patterns.",
    siteName: "Bitovi",
  },
  "https://www.bitovi.com/blog/kubernetes-deployment-strategies-explained": {
    title: "Kubernetes Deployment Strategies Explained",
    description:
      "Rolling, blue-green, and canary deployments on Kubernetes, and when to reach for each.",
    siteName: "Bitovi",
  },
  "https://www.bitovi.com/blog/infrastructure-as-code-with-terraform": {
    title: "Infrastructure as Code with Terraform",
    description:
      "Provision and version your cloud infrastructure with Terraform, modules, and remote state.",
    siteName: "Bitovi",
  },
  "https://www.bitovi.com/blog/building-ci-cd-pipelines-that-scale": {
    title: "Building CI/CD Pipelines That Scale",
    description:
      "Design CI/CD pipelines that stay fast as your team and codebase grow, with caching and parallelism.",
    siteName: "Bitovi",
  },
  "https://www.bitovi.com/blog/observability-logs-metrics-and-traces": {
    title: "Observability: Logs, Metrics, and Traces",
    description:
      "The three pillars of observability and how to instrument your services for production insight.",
    siteName: "Bitovi",
  },
  // count — AI tag listing
  "https://www.bitovi.com/blog/tags/ai": {
    title: "AI Articles on the Bitovi Blog",
    description:
      "Browse every Bitovi blog post tagged AI, covering LLMs, RAG, prompt engineering, and product integration.",
    siteName: "Bitovi",
  },
  // recency — latest post
  "https://www.bitovi.com/blog/server-components-in-practice-lessons-from-production": {
    title: "Server Components in Practice: Lessons from Production",
    description:
      "Real-world patterns, data-fetching trade-offs, and incremental migration tips for React Server Components.",
    siteName: "Bitovi",
  },
}

export function getMockPreview(target: URL): PreviewData {
  const stub = MOCK_PREVIEWS[target.toString()]
  if (stub) {
    return { ...stub, host: target.host }
  }
  return {
    title: target.pathname.split("/").filter(Boolean).pop()?.replace(/-/g, " ") ?? target.host,
    description: "Preview generated in mock mode.",
    siteName: "Bitovi",
    host: target.host,
  }
}
