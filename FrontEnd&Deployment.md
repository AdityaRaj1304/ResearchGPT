# Antigravity Staged Execution Plan: ResearchGPT Frontend & Deployment

> **CRITICAL AGENT DIRECTIVE:** Do NOT execute this plan in a single blind pass. You must start with **Stage 0 (Read-Only Repository Audit & Schema Alignment)** and report your findings before creating or modifying any files.

---

## Stage 0: Read-Only Repository Audit (EXECUTE FIRST — NO FILE MODIFICATIONS)

I want to extend my existing **ResearchGPT** project with a production-quality React + TypeScript frontend (`frontend/`) and eventually deploy the full stack.

Before making any changes, inspect the existing repository, specifically:
- `src/api/app.py`, `src/api/schemas.py`, and all backend endpoint implementations
- `src/models/` and `src/evaluation/`
- `src/config.py`
- `requirements.txt` and existing environment configuration
- ChromaDB persistence paths (`data/vector_store/` vs `chroma_db/`), index manifests, and actual `data/` directory sizes
- Existing tests (`tests/test_api_and_core.py`) and Git configuration

### Planned Frontend Modules
1. **Research Assistant**: Query indexed research papers, select a chunking collection and retrieval strategy, display generated answers with clickable inline citations (`[1]`, `[2]`), show retrieved source passages, and display actual retrieval/generation timings.
2. **Retrieval Lab**: Compare selected chunking collections and retrieval strategies using the existing `/ablation/compare` endpoint. Display retrieved paper IDs, source previews, and actual latency measurements.
3. **Benchmark Explorer**: Visualize real metrics from `/benchmarks/summary`, including Precision@K, Recall@K, MRR, nDCG@K, citation-support metrics where available, and statistical significance results.
4. **System Overview**: Display actual API health, loaded collections, vector counts, embedding dimensions, and LLM configuration status.

### Strict Engineering Requirements
- Use Vite, React, TypeScript, Tailwind CSS (pinned to v3.4 for deterministic configuration, or Vite's official `@tailwindcss/vite` v4 plugin), Lucide icons, and Recharts.
- Use a centralized, typed API client.
- **Derive frontend TypeScript interfaces strictly from the actual backend Pydantic schemas and return payloads**; do not invent response fields.
- Implement loading, empty, error, and unavailable-data states.
- Ensure citations link to the correct retrieved source.
- Keep API keys and secrets exclusively on the backend.
- Preserve existing backend architecture and retrieval algorithms.
- Do not start Docker or deployment work until the frontend and existing backend pass local verification.

### Deliverable for Stage 0 (STOP AND WAIT FOR APPROVAL AFTER THIS)
Return a structured report containing:
1. The **actual** backend endpoint schemas and exact JSON response shapes from `src/api/`.
2. The correct module import path and local startup command.
3. The actual ChromaDB persistence paths, SQLite/Parquet disk footprint, and deployment memory implications.
4. Any discrepancies between the reference frontend interfaces below and the real FastAPI backend.
5. Dependencies or build assumptions that could fail.
6. A phased implementation plan (Stage 1: Core Client + Research Assistant & Benchmarks $\rightarrow$ Stage 2: Retrieval Lab & System Overview $\rightarrow$ Stage 3: Docker & Deployment) with explicit acceptance checks.

**Wait for user approval before proceeding to Stage 1.**

---

## Stage 1 & 2: Reference Frontend Implementation (Corrected & Validated)

> **Note to Agent:** When implementing Stages 1 and 2 after approval, adapt the TypeScript interfaces in `src/types/api.ts` to match whatever exact field names you discovered in Stage 0. All syntax errors from earlier drafts have been fixed below.

### 1. Scaffold Vite + React + TypeScript with Pinned Tailwind CSS v3
To avoid Tailwind v4 `init` breaking changes, explicitly install `tailwindcss@^3.4.17`:

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install axios lucide-react recharts clsx tailwind-merge
npm install -D tailwindcss@^3.4.17 postcss autoprefixer @types/node
npx tailwindcss init -p
```

**File:** `frontend/tailwind.config.js`
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
};
```

**File:** `frontend/src/index.css`
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  @apply bg-slate-950 text-slate-100 antialiased selection:bg-indigo-500 selection:text-white;
}
```

---

### 2. Validated TypeScript Interfaces (`frontend/src/types/api.ts`)
> **Agent Action:** Cross-check every key below against `src/api/schemas.py` and `src/api/app.py` from your Stage 0 audit.

```typescript
export type CollectionType =
  | "arxiv_wholedoc"
  | "arxiv_fixed_512"
  | "arxiv_overlap_512_128"
  | "arxiv_semantic";

export type StrategyType = "dense" | "hybrid" | "hybrid_rerank";

export interface LatencyBreakdown {
  query_embedding_ms?: number;
  embedding_ms?: number;
  dense_search_ms?: number;
  dense_ms?: number;
  sparse_search_ms?: number;
  sparse_bm25_ms?: number;
  rrf_merge_ms?: number;
  rrf_fusion_ms?: number;
  rerank_ms?: number;
  generation_ms?: number;
  total_retrieval_ms?: number;
  total_ms?: number;
}

export interface RetrievedChunk {
  id: string;
  document: string;
  metadata: {
    paper_id?: string;
    title?: string;
    chunk_index?: number;
    token_count?: number;
    [key: string]: unknown;
  };
  score: number;
}

export interface QueryRequest {
  query: string;
  collection: CollectionType;
  strategy: StrategyType;
  top_k: number;
  generate_answer: boolean;
}

export interface QueryResponse {
  query: string;
  collection: string;
  strategy: string;
  top_k: number;
  answer: string | null;
  chunks: RetrievedChunk[];
  timings: LatencyBreakdown;
}

export interface CompareRequest {
  query: string;
  top_k: number;
  collections: CollectionType[];
  strategies: StrategyType[];
}

export interface ComparisonCell {
  collection: CollectionType;
  strategy: StrategyType;
  timings: LatencyBreakdown;
  top_chunk_preview: string;
  retrieved_paper_ids: string[];
  chunks?: RetrievedChunk[];
}

export interface CompareResponse {
  query: string;
  comparisons: ComparisonCell[];
}

export interface AggregatedMetricRow {
  collection: string;
  strategy: string;
  top_k: number;
  precision_at_k: number;
  recall_at_k: number;
  mrr: number;
  ndcg_at_k: number;
  CitationSupportRate?: number;
  UnsupportedClaimRate?: number;
  total_latency_ms?: number;
}

export interface StatisticalTestRow {
  comparison: string;
  metric: string;
  t_stat: number;
  p_value_ttest: number;
  wilcoxon_stat?: number;
  p_value_wilcoxon?: number;
  cohens_d?: number;
  significant: boolean;
}

export interface BenchmarkSummaryResponse {
  aggregated_metrics: AggregatedMetricRow[];
  statistical_tests: StatisticalTestRow[];
}

export interface HealthResponse {
  status: string;
  loaded_collections: string[];
  llm_configured: boolean;
}

export interface IndexStatRow {
  collection: string;
  vector_count: number;
  dimension: number;
  avg_tokens?: number;
  build_time_sec?: number;
  device?: string;
}

export interface CollectionsResponse {
  collections: string[];
  index_statistics: IndexStatRow[];
}
```

---

### 3. Centralized API Client (`frontend/src/api/client.ts`)
Supports both local development (`http://localhost:8000`), reverse-proxy relative routing (`/api`), and deployed cloud URLs via `VITE_API_URL`.

```typescript
import axios from "axios";
import {
  QueryRequest,
  QueryResponse,
  CompareRequest,
  CompareResponse,
  BenchmarkSummaryResponse,
  HealthResponse,
  CollectionsResponse,
} from "../types/api";

const API_BASE_URL =
  import.meta.env.VITE_API_URL?.trim() || "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
  headers: {
    "Content-Type": "application/json",
  },
});

export const researchApi = {
  getHealth: async (): Promise<HealthResponse> => {
    const { data } = await apiClient.get<HealthResponse>("/health");
    return data;
  },

  getCollections: async (): Promise<CollectionsResponse> => {
    const { data } = await apiClient.get<CollectionsResponse>("/collections");
    return data;
  },

  runQuery: async (payload: QueryRequest): Promise<QueryResponse> => {
    const { data } = await apiClient.post<QueryResponse>("/query", payload);
    return data;
  },

  compareAblation: async (payload: CompareRequest): Promise<CompareResponse> => {
    const { data } = await apiClient.post<CompareResponse>(
      "/ablation/compare",
      payload
    );
    return data;
  },

  getBenchmarkSummary: async (): Promise<BenchmarkSummaryResponse> => {
    const { data } = await apiClient.get<BenchmarkSummaryResponse>(
      "/benchmarks/summary"
    );
    return data;
  },
};
```

---

### 4. Module 1: Research Assistant (`frontend/src/components/ResearchAssistant.tsx`)
*Fixes applied:*
- Fixed state union type: `useState<QueryResponse | null>(null)`
- Fixed arXiv URL template literal: ``href={`https://arxiv.org/abs/${String(paperId).replace(/v\d+$/, "")}`}``

```tsx
import React, { useState, useRef } from "react";
import {
  Search,
  Sparkles,
  Clock,
  BookOpen,
  ExternalLink,
  Loader2,
} from "lucide-react";
import { researchApi } from "../api/client";
import { CollectionType, StrategyType, QueryResponse } from "../types/api";

export const ResearchAssistant: React.FC = () => {
  const [query, setQuery] = useState(
    "How does semantic chunking compare to fixed-size token windows for long-context retrieval?"
  );
  const [collection, setCollection] = useState<CollectionType>("arxiv_semantic");
  const [strategy, setStrategy] = useState<StrategyType>("hybrid_rerank");
  const [topK, setTopK] = useState<number>(5);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse null |>(null);
  const [activeChunkIndex, setActiveChunkIndex] = useState<number | null>(null);

  const chunkRefs = useRef<(HTMLDivElement | null)[]>([]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setActiveChunkIndex(null);

    try {
      const res = await researchApi.runQuery({
        query,
        collection,
        strategy,
        top_k: topK,
        generate_answer: true,
      });
      setResult(res);
    } catch (err: any) {
      setError(
        err?.response?.data?.detail ||
          "Failed to execute query against backend. Verify the FastAPI server is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleCitationClick = (index: number) => {
    const targetIdx = index - 1;
    if (result && targetIdx >= 0 && targetIdx < result.chunks.length) {
      setActiveChunkIndex(targetIdx);
      chunkRefs.current[targetIdx]?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  };

  const renderCitedAnswer = (text: string) => {
    const parts = text.split(/(\[\d+\])/g);
    return parts.map((part, idx) => {
      const match = part.match(/^\[(\d+)\]$/);
      if (match) {
        const citationNum = parseInt(match[1], 10);
        return (
          <button
            key={idx}
            type="button"
            onClick={() => handleCitationClick(citationNum)}
            className="mx-1 inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 hover:bg-indigo-500 hover:text-white transition"
            title={`Jump to Source Passage [${citationNum}]`}
          >
            [{citationNum}]
          </button>
        );
      }
      return <span key={idx}>{part}</span>;
    });
  };

  return (
    <div className="space-y-6">
      <form
        onSubmit={handleSearch}
        className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4 shadow-lg"
      >
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a research question over indexed arXiv papers..."
            className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-11 pr-32 py-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
          />
          <Search className="w-5 h-5 text-slate-400 absolute left-3.5 top-4"/>
          <button
            type="submit"
            disabled={loading}
            className="absolute right-2 top-2 bottom-2 px-5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium rounded-md flex items-center gap-2 transition"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin"/>
            ) : (
              <Sparkles className="w-4 h-4"/>
            )}
            Ask
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-1">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">
              Chunking Strategy (Collection)
            </label>
            <select
              value={collection}
              onChange={(e) => setCollection(e.target.value as CollectionType)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200"
            >
              <option value="arxiv_semantic">
                Semantic Bounded (Cosine 20th Pct)
              </option>
              <option value="arxiv_overlap_512_128">
                Overlap (512 tok / 128 stride)
              </option>
              <option value="arxiv_fixed_512">
                Fixed-Size (512 tok / 0 overlap)
              </option>
              <option value="arxiv_wholedoc">Whole-Document Baseline</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">
              Retrieval Architecture
            </label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as StrategyType)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200"
            >
              <option value="hybrid_rerank">
                Hybrid (Dense + BM25 + RRF) + Cross-Encoder
              </option>
              <option value="hybrid">Hybrid (Dense + BM25 + RRF k=60)</option>
              <option value="dense">Dense Only (BGE-Small Cosine)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">
              Top-K Context Cutoff
            </label>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200"
            >
              <option value={3}>Top-3 Chunks</option>
              <option value={5}>Top-5 Chunks</option>
              <option value={10}>Top-10 Chunks</option>
            </select>
          </div>
        </div>
      </form>

      {error && (
        <div className="bg-red-950/50 border border-red-800 text-red-200 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7 space-y-4">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-indigo-400"/>
                  <h3 className="font-semibold text-slate-100">
                    Grounded Literature Synthesis (Llama-3.3-70B)
                  </h3>
                </div>
                <span className="text-xs font-mono text-slate-400">
                  temp = 0.0
                </span>
              </div>

              <div className="prose prose-invert max-w-none text-slate-200 leading-relaxed whitespace-pre-line text-sm">
                {result.answer
                  ? renderCitedAnswer(result.answer)
                  : "No LLM synthesis generated."}
              </div>

              <div className="pt-4 border-t border-slate-800/80 flex flex-wrap gap-2 items-center text-xs font-mono">
                <span className="flex items-center gap-1 text-slate-400 mr-2">
                  <Clock className="w-3.5 h-3.5 text-indigo-400"/> Telemetry:
                </span>
                <span className="bg-slate-950 border border-slate-800 px-2.5 py-1 rounded text-slate-300">
                  Embed:{" "}
                  {(
                    result.timings.query_embedding_ms ??
                    result.timings.embedding_ms ??
                    0
                  ).toFixed(2)}
                  ms
                </span>
                <span className="bg-slate-950 border border-slate-800 px-2.5 py-1 rounded text-slate-300">
                  Dense:{" "}
                  {(
                    result.timings.dense_search_ms ??
                    result.timings.dense_ms ??
                    0
                  ).toFixed(2)}
                  ms
                </span>
                <span className="bg-slate-950 border border-slate-800 px-2.5 py-1 rounded text-slate-300">
                  BM25:{" "}
                  {(
                    result.timings.sparse_search_ms ??
                    result.timings.sparse_bm25_ms ??
                    0
                  ).toFixed(2)}
                  ms
                </span>
                <span className="bg-slate-950 border border-slate-800 px-2.5 py-1 rounded text-amber-300">
                  Rerank: {(result.timings.rerank_ms ?? 0).toFixed(2)}ms
                </span>
                <span className="bg-indigo-950/60 border border-indigo-700/50 px-2.5 py-1 rounded text-indigo-300">
                  Total:{" "}
                  {(
                    result.timings.total_ms ??
                    result.timings.total_retrieval_ms ??
                    0
                  ).toFixed(1)}
                  ms
                </span>
              </div>
            </div>
          </div>

          <div className="lg:col-span-5 space-y-3">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-indigo-400"/>
              Retrieved Source Passages ({result.chunks.length})
            </h3>

            <div className="space-y-3 max-h-[640px] overflow-y-auto pr-1">
              {result.chunks.map((chunk, i) => {
                const isHighlighted = activeChunkIndex === i;
                const paperId = String(chunk.metadata?.paper_id || "Unknown");
                const cleanArxivId = paperId.replace(/v\d+$/, "");
                return (
                  <div
                    key={chunk.id || i}
                    ref={(el) => {
                      chunkRefs.current[i] = el;
                    }}
                    onClick={() => setActiveChunkIndex(i)}
                    className={`cursor-pointer rounded-xl p-4 border transition ${
                      isHighlighted
                        ? "bg-indigo-950/40 border-indigo-500 ring-2 ring-indigo-500/40"
                        : "bg-slate-900/80 border-slate-800 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-0.5 text-xs font-mono font-bold rounded bg-indigo-600 text-white">
                          [{i + 1}]
                        </span>
                        <span className="text-xs font-mono text-slate-300">
                          arXiv:{paperId}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-emerald-400 bg-emerald-950/50 border border-emerald-800/60 px-2 py-0.5 rounded">
                          score: {Number(chunk.score ?? 0).toFixed(4)}
                        </span>
                        {paperId !== "Unknown" && (
                          <a
                            href={`[https://arxiv.org/abs/$](https://arxiv.org/abs/$){cleanArxivId}`}
                            target="_blank"
                            rel="noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="text-slate-400 hover:text-indigo-400"
                            title="Open paper on arXiv"
                          >
                            <ExternalLink className="w-3.5 h-3.5"/>
                          </a>
                        )}
                      </div>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed font-sans">
                      {chunk.document}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
```

---

### 5. Module 3: Benchmark Explorer (`frontend/src/components/BenchmarkExplorer.tsx`)
*Fixes applied:*
- Repaired corrupted `<BarChart>`, `<XAxis>`, `<YAxis>`, `<Tooltip>`, and `<Bar>` JSX attributes.
- Added error and empty-data fallback states.

```tsx
import React, { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { BarChart3, Award, CheckCircle2, XCircle, AlertCircle } from "lucide-react";
import { researchApi } from "../api/client";
import { AggregatedMetricRow, StatisticalTestRow } from "../types/api";

export const BenchmarkExplorer: React.FC = () => {
  const [metrics, setMetrics] = useState<AggregatedMetricRow[]>([]);
  const [stats, setStats] = useState<StatisticalTestRow[]>([]);
  const [topKFilter, setTopKFilter] = useState<number>(5);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    researchApi
      .getBenchmarkSummary()
      .then((data) => {
        setMetrics(data.aggregated_metrics || []);
        setStats(data.statistical_tests || []);
      })
      .catch((err: any) => {
        setError(
          err?.response?.data?.detail ||
            "Could not load benchmark results from /benchmarks/summary."
        );
      })
      .finally(() => setLoading(false));
  }, []);

  const filteredMetrics = metrics
    .filter((m) => Number(m.top_k) === topKFilter)
    .map((m) => ({
      ...m,
      label: `${m.collection.replace("arxiv_", "")} (${m.strategy})`,
    }));

  if (loading) {
    return (
      <div className="text-slate-400 text-sm p-8 bg-slate-900/50 border border-slate-800 rounded-xl">
        Loading benchmark telemetry...
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-950/50 border border-red-800 text-red-200 p-4 rounded-xl flex items-center gap-2 text-sm">
        <AlertCircle className="w-4 h-4 shrink-0"/>
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between bg-slate-900 border border-slate-800 rounded-xl p-4">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-indigo-400"/>
          <h2 className="font-semibold text-slate-100">
            24-Configuration Ablation Benchmark Explorer
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">Filter Cutoff:</span>
          {[3, 5].map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => setTopKFilter(k)}
              className={`px-3 py-1 rounded text-xs font-mono ${
                topKFilter === k
                  ? "bg-indigo-600 text-white"
                  : "bg-slate-950 text-slate-400 border border-slate-800"
              }`}
            >
              Top-K = {k}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-slate-200 mb-4">
          nDCG@{topKFilter}, MRR, Precision@{topKFilter}, and Recall@{topKFilter} by Configuration
        </h3>
        <div className="h-80 w-full">
          <ResponsiveContainer height="100%" width="100%">
            <BarChart 10 20, 60, bottom: data="{filteredMetrics}" left: margin="{{" right: top: }}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3"/>
              <XAxis angle="{-30}" dataKey="label" fontSize="{11}" interval="{0}" stroke="#94a3b8" textAnchor="end"/>
              <YAxis 1]} domain="{[0," fontSize="{12}" stroke="#94a3b8"/>
              <Tooltip "#0f172a", "#334155", "12px", backgroundColor: borderColor: contentStyle="{{" fontSize: }}/>
              <Legend verticalAlign="top"/>
              <Bar dataKey="ndcg_at_k" fill="#6366f1" name="{`nDCG@${topKFilter}`}"/>
              <Bar dataKey="mrr" fill="#10b981" name="MRR"/>
              <Bar dataKey="precision_at_k" fill="#38bdf8" name="{`P@${topKFilter}`}"/>
              <Bar dataKey="recall_at_k" fill="#f59e0b" name="{`R@${topKFilter}`}"/>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
        <div className="flex items-center gap-2">
          <Award className="w-5 h-5 text-emerald-400"/>
          <h3 className="text-sm font-semibold text-slate-100">
            Paired Hypothesis Tests (Student's t-test, Wilcoxon Signed-Rank & Cohen's d)
          </h3>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-mono">
                <th className="py-2.5 px-3">Comparison Pair</th>
                <th className="py-2.5 px-3">Metric</th>
                <th className="py-2.5 px-3">t-Stat</th>
                <th className="py-2.5 px-3">p-Value (t-test)</th>
                <th className="py-2.5 px-3">p-Value (Wilcoxon)</th>
                <th className="py-2.5 px-3">Cohen's d</th>
                <th className="py-2.5 px-3">Significant (α=0.05)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {stats.map((row, i) => (
                <tr key={i} className="hover:bg-slate-950/60">
                  <td className="py-2.5 px-3 text-slate-200">{row.comparison}</td>
                  <td className="py-2.5 px-3 text-indigo-300">{row.metric}</td>
                  <td className="py-2.5 px-3">
                    {Number(row.t_stat ?? 0).toFixed(3)}
                  </td>
                  <td className="py-2.5 px-3">
                    {Number(row.p_value_ttest ?? 1).toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3">
                    {row.p_value_wilcoxon !== undefined
                      ? Number(row.p_value_wilcoxon).toFixed(4)
                      : "N/A"}
                  </td>
                  <td className="py-2.5 px-3 text-amber-300">
                    {row.cohens_d !== undefined
                      ? Number(row.cohens_d).toFixed(3)
                      : "N/A"}
                  </td>
                  <td className="py-2.5 px-3">
                    {row.significant ? (
                      <span className="inline-flex items-center gap-1 text-emerald-400">
                        <CheckCircle2 className="w-3.5 h-3.5"/> Yes
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-slate-500">
                        <XCircle className="w-3.5 h-3.5"/> No
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
```

---

### 6. Module 4: System Overview (`frontend/src/components/SystemOverview.tsx`)
*Fixes applied:*
- Corrected union types: `useState<HealthResponse | null>(null)` and `useState<CollectionsResponse | null>(null)`

```tsx
import React, { useEffect, useState } from "react";
import {
  Server,
  Database,
  Cpu,
  CheckCircle,
  AlertTriangle,
} from "lucide-react";
import { researchApi } from "../api/client";
import { HealthResponse, CollectionsResponse } from "../types/api";

export const SystemOverview: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse null |>(null);
  const [collectionsData, setCollectionsData] =
    useState<CollectionsResponse null |>(null);

  useEffect(() => {
    researchApi
      .getHealth()
      .then(setHealth)
      .catch(() => null);
    researchApi
      .getCollections()
      .then(setCollectionsData)
      .catch(() => null);
  }, []);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
          <div>
            <span className="text-xs uppercase text-slate-400">
              FastAPI Service Status
            </span>
            <h3 className="text-lg font-bold text-slate-100 mt-1">
              {health?.status === "healthy" ? "Operational" : "Connecting..."}
            </h3>
            <span className="text-xs text-slate-400">
              Lifespan Cache: {health?.loaded_collections?.length ?? 0} / 4 Collections
            </span>
          </div>
          <Server className="w-8 h-8 text-emerald-400"/>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
          <div>
            <span className="text-xs uppercase text-slate-400">
              Vector & Rerank Stack
            </span>
            <h3 className="text-sm font-mono font-semibold text-indigo-300 mt-1">
              BAAI/bge-small-en-v1.5 (384d)
            </h3>
            <span className="text-xs font-mono text-slate-400">
              Reranker: ms-marco-MiniLM-L-6-v2
            </span>
          </div>
          <Cpu className="w-8 h-8 text-indigo-400"/>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
          <div>
            <span className="text-xs uppercase text-slate-400">
              Synthesis Engine (Groq)
            </span>
            <h3 className="text-sm font-mono font-semibold text-slate-100 mt-1">
              llama-3.3-70b-versatile
            </h3>
            <span className="text-xs flex items-center gap-1 mt-1">
              {health?.llm_configured ? (
                <span className="text-emerald-400 flex items-center gap-1">
                  <CheckCircle className="w-3.5 h-3.5"/> API Key Configured
                </span>
              ) : (
                <span className="text-amber-400 flex items-center gap-1">
                  <AlertTriangle className="w-3.5 h-3.5"/> Key Missing
                </span>
              )}
            </span>
          </div>
          <Database className="w-8 h-8 text-amber-400"/>
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-slate-100">
          ChromaDB HNSW Vector Collections & Index Telemetry
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-3">Collection Name</th>
                <th className="py-2.5 px-3">Vector Count</th>
                <th className="py-2.5 px-3">Embedding Dim</th>
                <th className="py-2.5 px-3">Similarity Metric</th>
                <th className="py-2.5 px-3">In-Memory BM25 Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {(collectionsData?.index_statistics?.length
                ? collectionsData.index_statistics
                : (collectionsData?.collections || []).map((c) => ({
                    collection: c,
                    vector_count: 0,
                    dimension: 384,
                  }))
              ).map((item, idx) => (
                <tr key={idx} className="hover:bg-slate-950/50">
                  <td className="py-2.5 px-3 text-indigo-300 font-semibold">
                    {item.collection}
                  </td>
                  <td className="py-2.5 px-3 text-slate-200">
                    {item.vector_count || "Indexed"}
                  </td>
                  <td className="py-2.5 px-3 text-slate-300">
                    {item.dimension || 384}
                  </td>
                  <td className="py-2.5 px-3 text-slate-300">Cosine (HNSW)</td>
                  <td className="py-2.5 px-3 text-emerald-400">Warm / Ready</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
```

---

## Stage 3: Production Reverse Proxy & Deployment (Execute ONLY After Local Verification Passes)

### Production API Routing Fix (`frontend/nginx.conf`)
Instead of hardcoding `http://localhost:8000` in production builds, configure Nginx inside the frontend container to serve the React SPA and proxy `/api/*` requests directly to the backend service, or pass `VITE_API_URL` at build time:

```nginx
server {
    listen 80;
    server_name _;

    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://backend:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
```

### Verification Gate Before Deployment
1. Run `pytest tests/test_api_and_core.py -v` and confirm 100% pass rate.
2. Run `cd frontend && npx tsc --noEmit && npm run build` to verify zero TypeScript or JSX compilation errors.
3. Start the FastAPI server and Vite dev server locally and verify that `/health`, `/collections`, `/benchmarks/summary`, `/query`, and `/ablation/compare` populate all 4 tabs without console or network errors.