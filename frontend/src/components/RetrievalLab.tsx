import React, { useState } from "react";
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
import {
  Layers,
  Zap,
  Clock,
  ExternalLink,
  AlertCircle,
  HelpCircle,
  BarChart2,
  FileText,
  Sliders,
  Sparkles,
  Loader2,
  Check,
} from "lucide-react";
import { researchApi } from "../api/client";
import type {
  CollectionType,
  StrategyType,
  CompareResponse,
  ComparisonCell,
} from "../types/api";

interface CollectionOption {
  id: CollectionType;
  label: string;
  description: string;
}

interface StrategyOption {
  id: StrategyType;
  label: string;
  description: string;
  color: string;
}

const AVAILABLE_COLLECTIONS: CollectionOption[] = [
  {
    id: "arxiv_wholedoc",
    label: "Whole Document Baseline",
    description: "Full un-chunked paper text (1 vector/doc)",
  },
  {
    id: "arxiv_fixed_512",
    label: "Fixed 512 (0 Overlap)",
    description: "Strict 512-token chunks without sliding overlap",
  },
  {
    id: "arxiv_overlap_512_128",
    label: "Sliding 512 / 128",
    description: "512-token chunks with 128-token boundary overlap",
  },
  {
    id: "arxiv_semantic_512",
    label: "Semantic 512",
    description: "Dynamic semantic boundary chunking (~512 target)",
  },
];

const AVAILABLE_STRATEGIES: StrategyOption[] = [
  {
    id: "dense",
    label: "Dense (BGE-Small)",
    description: "Bi-encoder cosine similarity via ChromaDB HNSW",
    color: "text-sky-400 border-sky-500/30 bg-sky-500/10",
  },
  {
    id: "hybrid",
    label: "Hybrid (BM25 + Dense + RRF)",
    description: "Rank-fusion of BM25 sparse scores & dense vectors",
    color: "text-purple-400 border-purple-500/30 bg-purple-500/10",
  },
  {
    id: "hybrid_rerank",
    label: "Hybrid + Cross-Encoder",
    description: "Hybrid candidates re-ranked by ms-marco-MiniLM-L-6-v2",
    color: "text-amber-400 border-amber-500/30 bg-amber-500/10",
  },
];

const SAMPLE_QUERIES = [
  "How does semantic chunking compare to sliding-window chunking for dense retrieval?",
  "Cross-encoder reranking improvements on vocabulary mismatch in academic papers",
  "Reciprocal Rank Fusion k parameter tuning for BM25 and dense bi-encoder",
];

const formatCollectionLabel = (id: string): string => {
  switch (id) {
    case "arxiv_wholedoc":
      return "Whole Doc";
    case "arxiv_fixed_512":
      return "Fixed 512";
    case "arxiv_overlap_512_128":
      return "Sliding 512/128";
    case "arxiv_semantic_512":
    case "arxiv_semantic":
      return "Semantic 512";
    default:
      return id.replace("arxiv_", "");
  }
};

const formatStrategyLabel = (strat: string): string => {
  switch (strat) {
    case "dense":
      return "Dense";
    case "hybrid":
      return "Hybrid (RRF)";
    case "hybrid_rerank":
      return "Hybrid + Rerank";
    default:
      return strat;
  }
};

export const RetrievalLab: React.FC = () => {
  const [query, setQuery] = useState(
    "How does semantic chunking compare to sliding-window chunking for dense retrieval?"
  );
  const [selectedCollections, setSelectedCollections] = useState<CollectionType[]>([
    "arxiv_fixed_512",
    "arxiv_semantic_512",
  ]);
  const [selectedStrategies, setSelectedStrategies] = useState<StrategyType[]>([
    "dense",
    "hybrid_rerank",
  ]);
  const [topK, setTopK] = useState<number>(5);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<CompareResponse | null>(null);

  const toggleCollection = (id: CollectionType) => {
    setSelectedCollections((prev) =>
      prev.includes(id) ? prev.filter((c) => c !== id) : [...prev, id]
    );
  };

  const toggleStrategy = (id: StrategyType) => {
    setSelectedStrategies((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]
    );
  };

  const handleSelectAllCollections = () => {
    setSelectedCollections(AVAILABLE_COLLECTIONS.map((c) => c.id));
  };

  const handleSelectAllStrategies = () => {
    setSelectedStrategies(AVAILABLE_STRATEGIES.map((s) => s.id));
  };

  const handleCompare = async (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim().length < 3) {
      setError("Search query must be at least 3 characters long.");
      return;
    }
    if (selectedCollections.length === 0) {
      setError("Please select at least one chunking collection.");
      return;
    }
    if (selectedStrategies.length === 0) {
      setError("Please select at least one retrieval strategy.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const resp = await researchApi.compareAblation({
        query: query.trim(),
        top_k: topK,
        collections: selectedCollections,
        strategies: selectedStrategies,
      });
      setResults(resp);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(
        typeof detail === "string"
          ? detail
          : "Failed to complete ablation comparison. Ensure backend is running and collections are populated."
      );
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data if results exist
  const chartData = results?.comparisons.map((c) => ({
    label: `${formatCollectionLabel(c.collection)} · ${formatStrategyLabel(c.strategy)}`,
    retrieval_ms:
      c.timings.total_retrieval_ms !== undefined
        ? Number(c.timings.total_retrieval_ms.toFixed(2))
        : null,
    total_ms:
      c.timings.total_ms !== undefined
        ? Number(c.timings.total_ms.toFixed(2))
        : null,
  })) || [];

  const totalConfigurations =
    selectedCollections.length * selectedStrategies.length;

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2">
          <h2 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <Layers className="w-5 h-5 text-indigo-400" />
            Retrieval Ablation Lab
          </h2>
          <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
            POST /ablation/compare
          </span>
        </div>
        <p className="text-xs text-slate-400 mt-1 max-w-3xl leading-relaxed">
          Perform multi-parameter ablation across ChromaDB chunking collections and
          retrieval architectures. Inspect side-by-side latency breakdowns, pure
          retrieval time versus total endpoint latency, retrieved paper IDs, and
          top-chunk text snippets returned directly by the backend.
        </p>
      </div>

      {/* Configuration & Input Form */}
      <form
        onSubmit={handleCompare}
        className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur space-y-6"
      >
        {/* Query Input */}
        <div className="space-y-2">
          <label className="text-xs font-semibold text-slate-300 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
              Academic Research Query
            </span>
            <span className="text-[11px] font-mono text-slate-500">
              Min 3 characters
            </span>
          </label>
          <div className="relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. How does semantic chunking compare to fixed-size token windows?"
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
            />
          </div>

          {/* Sample Query Chips */}
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <span className="text-[11px] text-slate-500 flex items-center gap-1">
              Sample queries:
            </span>
            {SAMPLE_QUERIES.map((sq, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setQuery(sq)}
                className="text-[11px] px-2.5 py-1 rounded-lg bg-slate-800/60 hover:bg-slate-800 text-slate-300 border border-slate-700/60 hover:border-slate-600 transition text-left cursor-pointer"
              >
                {sq}
              </button>
            ))}
          </div>
        </div>

        {/* Collection & Strategy Selectors Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2 border-t border-slate-800/80">
          {/* Collection Selection */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5 text-indigo-400" />
                Chunking Collections ({selectedCollections.length}/{AVAILABLE_COLLECTIONS.length})
              </label>
              <div className="flex items-center gap-2 text-[11px]">
                <button
                  type="button"
                  onClick={handleSelectAllCollections}
                  className="text-indigo-400 hover:text-indigo-300 transition"
                >
                  Select All
                </button>
                <span className="text-slate-600">·</span>
                <button
                  type="button"
                  onClick={() => setSelectedCollections([])}
                  className="text-slate-400 hover:text-slate-300 transition"
                >
                  Clear
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {AVAILABLE_COLLECTIONS.map((col) => {
                const isSelected = selectedCollections.includes(col.id);
                return (
                  <button
                    key={col.id}
                    type="button"
                    onClick={() => toggleCollection(col.id)}
                    className={`p-3 rounded-xl border text-left transition flex items-start gap-2.5 cursor-pointer ${
                      isSelected
                        ? "bg-indigo-950/30 border-indigo-500/50 text-slate-100"
                        : "bg-slate-950/60 border-slate-800/80 text-slate-400 hover:border-slate-700"
                    }`}
                  >
                    <div
                      className={`w-4 h-4 rounded mt-0.5 flex items-center justify-center border transition ${
                        isSelected
                          ? "bg-indigo-600 border-indigo-500 text-white"
                          : "border-slate-700 bg-slate-900"
                      }`}
                    >
                      {isSelected && <Check className="w-3 h-3 stroke-[3]" />}
                    </div>
                    <div>
                      <div className="text-xs font-medium text-slate-200">
                        {col.label}
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5 leading-snug">
                        {col.description}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Strategy Selection */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-purple-400" />
                Retrieval Strategies ({selectedStrategies.length}/{AVAILABLE_STRATEGIES.length})
              </label>
              <div className="flex items-center gap-2 text-[11px]">
                <button
                  type="button"
                  onClick={handleSelectAllStrategies}
                  className="text-indigo-400 hover:text-indigo-300 transition"
                >
                  Select All
                </button>
                <span className="text-slate-600">·</span>
                <button
                  type="button"
                  onClick={() => setSelectedStrategies([])}
                  className="text-slate-400 hover:text-slate-300 transition"
                >
                  Clear
                </button>
              </div>
            </div>

            <div className="space-y-2">
              {AVAILABLE_STRATEGIES.map((strat) => {
                const isSelected = selectedStrategies.includes(strat.id);
                return (
                  <button
                    key={strat.id}
                    type="button"
                    onClick={() => toggleStrategy(strat.id)}
                    className={`w-full p-2.5 rounded-xl border text-left transition flex items-center justify-between cursor-pointer ${
                      isSelected
                        ? "bg-slate-800/70 border-indigo-500/50 text-slate-100"
                        : "bg-slate-950/60 border-slate-800/80 text-slate-400 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <div
                        className={`w-4 h-4 rounded flex items-center justify-center border transition ${
                          isSelected
                            ? "bg-indigo-600 border-indigo-500 text-white"
                            : "border-slate-700 bg-slate-900"
                        }`}
                      >
                        {isSelected && <Check className="w-3 h-3 stroke-[3]" />}
                      </div>
                      <div>
                        <div className="text-xs font-medium text-slate-200">
                          {strat.label}
                        </div>
                        <div className="text-[10px] text-slate-400">
                          {strat.description}
                        </div>
                      </div>
                    </div>
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded border ${strat.color}`}
                    >
                      {strat.id}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Bottom Bar: Top-K & Action Button */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-slate-800/80">
          <div className="flex items-center gap-4 w-full sm:w-auto">
            <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 whitespace-nowrap">
              <Sliders className="w-3.5 h-3.5 text-indigo-400" />
              Top-K Chunks:
            </label>
            <div className="flex items-center gap-2 flex-1 sm:flex-initial">
              <input
                type="range"
                min="1"
                max="10"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-28 sm:w-36 accent-indigo-500 cursor-pointer"
              />
              <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-800 text-indigo-300 font-bold border border-slate-700">
                {topK}
              </span>
              <span className="text-[11px] text-slate-500">(1–10)</span>
            </div>
          </div>

          <div className="flex items-center gap-3 w-full sm:w-auto justify-end">
            <span className="text-[11px] font-mono text-slate-400 hidden lg:inline">
              {totalConfigurations} configuration{totalConfigurations === 1 ? "" : "s"} selected
            </span>

            <button
              type="submit"
              disabled={loading || totalConfigurations === 0 || query.trim().length < 3}
              className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-medium text-xs sm:text-sm shadow-lg shadow-indigo-600/20 transition cursor-pointer disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Ablating Pipelines...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Run Ablation Comparison
                </>
              )}
            </button>
          </div>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-500/40 text-rose-200 text-xs flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-semibold text-rose-300">Ablation Error</div>
              <p className="text-rose-200/90 leading-relaxed">{error}</p>
            </div>
          </div>
        )}
      </form>

      {/* Scientific Interpretation Callout */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
        <div className="flex items-center gap-2 text-indigo-300 text-xs font-semibold">
          <HelpCircle className="w-4 h-4 text-indigo-400" />
          How to Interpret Retrieval Ablation Results
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-400 leading-relaxed">
          <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-1">
            <div className="font-medium text-slate-200 flex items-center gap-1.5">
              <Clock className="w-3 h-3 text-sky-400" />
              Retrieval vs. Total Latency
            </div>
            <p className="text-[11px]">
              <strong className="text-slate-300">Retrieval Latency</strong> measures pure
              internal execution (embedding generation, vector indexing, BM25 scoring,
              and cross-encoder reranking). <strong className="text-slate-300">Total Latency</strong> includes
              FastAPI HTTP deserialization and serialization overhead.
            </p>
          </div>

          <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-1">
            <div className="font-medium text-slate-200 flex items-center gap-1.5">
              <AlertCircle className="w-3 h-3 text-amber-400" />
              Speed Does Not Imply Accuracy
            </div>
            <p className="text-[11px]">
              Dense retrieval runs fastest because it skips BM25 inverted index lookups
              and pairwise cross-attention. However, higher speed does not guarantee better
              relevance. Consult the <strong className="text-slate-300">Benchmark Explorer</strong> to inspect
              empirical nDCG@5 and recall@5 metrics.
            </p>
          </div>

          <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-1">
            <div className="font-medium text-slate-200 flex items-center gap-1.5">
              <Zap className="w-3 h-3 text-purple-400" />
              Cross-Encoder Overhead
            </div>
            <p className="text-[11px]">
              The cross-encoder reranker (<code className="text-amber-300 font-mono text-[10px]">ms-marco-MiniLM-L-6-v2</code>)
              performs full cross-attention over query-passage token pairs. While it incurs
              2×–8× higher retrieval latency, it statistically eliminates false-positive dense matches.
            </p>
          </div>
        </div>
      </div>

      {/* Loading State Banner for Slow Requests */}
      {loading && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-10 text-center space-y-4 animate-pulse">
          <div className="p-3 w-12 h-12 mx-auto rounded-xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 flex items-center justify-center">
            <Loader2 className="w-6 h-6 animate-spin" />
          </div>
          <div className="space-y-1">
            <h3 className="text-sm font-bold text-slate-200">
              Running Ablation Across {totalConfigurations} Pipeline Configurations
            </h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
              Query embedding, vector retrieval, and cross-encoder scoring are executing sequentially
              on the backend. This may take 2–6 seconds depending on the number of rerank passes.
            </p>
          </div>
        </div>
      )}

      {/* Results Display */}
      {results && !loading && (
        <div className="space-y-6">
          {/* Summary Bar */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/60 border border-slate-800">
            <div>
              <span className="text-xs text-slate-400">Comparison Query: </span>
              <span className="text-xs font-semibold text-slate-200">"{results.query}"</span>
            </div>
            <div className="flex items-center gap-3 text-xs font-mono">
              <span className="px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
                Top-K: {results.top_k ?? topK}
              </span>
              <span className="px-2.5 py-1 rounded bg-indigo-950 text-indigo-300 border border-indigo-800/60 font-semibold">
                {results.comparisons.length} Pipelines Compared
              </span>
            </div>
          </div>

          {/* Visual Latency Chart */}
          {chartData.length > 0 && (
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <BarChart2 className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-bold text-slate-200">
                    Latency Comparison (ms)
                  </h3>
                </div>
                <div className="text-[11px] text-slate-500 font-mono">
                  Pure Retrieval vs. Total Round-Trip Latency
                </div>
              </div>

              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="label"
                      stroke="#64748b"
                      fontSize={11}
                      angle={-15}
                      textAnchor="end"
                      interval={0}
                    />
                    <YAxis
                      stroke="#64748b"
                      fontSize={11}
                      tickFormatter={(val) => `${val}ms`}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        borderColor: "#334155",
                        borderRadius: "0.75rem",
                        fontSize: "12px",
                      }}
                      formatter={(value: any) =>
                        value !== null && value !== undefined ? [`${value} ms`] : ["Unavailable"]
                      }
                    />
                    <Legend
                      verticalAlign="top"
                      height={36}
                      wrapperStyle={{ fontSize: "12px" }}
                    />
                    <Bar
                      dataKey="retrieval_ms"
                      name="Pure Retrieval Latency"
                      fill="#6366f1"
                      radius={[4, 4, 0, 0]}
                    />
                    <Bar
                      dataKey="total_ms"
                      name="Total Latency (incl. HTTP Serialization)"
                      fill="#38bdf8"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Side-by-Side Comparison Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {results.comparisons.map((comp: ComparisonCell, idx: number) => {
              const stratMeta = AVAILABLE_STRATEGIES.find((s) => s.id === comp.strategy);
              const retrMs = comp.timings.total_retrieval_ms;
              const totalMs = comp.timings.total_ms;

              return (
                <div
                  key={`${comp.collection}-${comp.strategy}-${idx}`}
                  className="bg-slate-900/90 border border-slate-800 hover:border-slate-700/80 rounded-2xl p-5 shadow-lg flex flex-col justify-between space-y-4 transition"
                >
                  {/* Card Header: Collection & Strategy */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-100 flex items-center gap-1.5">
                        <Layers className="w-3.5 h-3.5 text-indigo-400" />
                        {formatCollectionLabel(comp.collection)}
                      </span>
                      <span
                        className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                          stratMeta?.color || "text-slate-300 border-slate-700 bg-slate-800"
                        }`}
                      >
                        {formatStrategyLabel(comp.strategy)}
                      </span>
                    </div>
                    <div className="text-[11px] font-mono text-slate-500 truncate">
                      {comp.collection}
                    </div>
                  </div>

                  {/* Latency Comparison Box */}
                  <div className="bg-slate-950/80 rounded-xl p-3.5 border border-slate-800/80 space-y-3">
                    <div className="grid grid-cols-2 gap-2 text-center divide-x divide-slate-800">
                      <div>
                        <div className="text-[10px] text-slate-400 uppercase font-mono">
                          Pure Retrieval
                        </div>
                        <div className="text-base font-bold text-indigo-400 font-mono mt-0.5">
                          {retrMs !== undefined ? `${retrMs.toFixed(2)} ms` : "—"}
                        </div>
                      </div>
                      <div className="pl-2">
                        <div className="text-[10px] text-slate-400 uppercase font-mono">
                          Total Latency
                        </div>
                        <div className="text-base font-bold text-sky-400 font-mono mt-0.5">
                          {totalMs !== undefined ? `${totalMs.toFixed(2)} ms` : "—"}
                        </div>
                      </div>
                    </div>

                    {/* Available Stage Timings */}
                    <div className="pt-2 border-t border-slate-800/80 space-y-1 text-[11px] font-mono">
                      {comp.timings.query_embedding_ms !== undefined && (
                        <div className="flex justify-between text-slate-400">
                          <span>Query Embedding:</span>
                          <span className="text-slate-200 font-semibold">
                            {comp.timings.query_embedding_ms.toFixed(2)} ms
                          </span>
                        </div>
                      )}

                      {comp.timings.dense_search_ms !== undefined && (
                        <div className="flex justify-between text-slate-400">
                          <span>Dense Search:</span>
                          <span className="text-slate-200 font-semibold">
                            {comp.timings.dense_search_ms.toFixed(2)} ms
                          </span>
                        </div>
                      )}

                      {comp.timings.sparse_search_ms !== undefined && (
                        <div className="flex justify-between text-slate-400">
                          <span>Sparse BM25:</span>
                          <span className="text-slate-200 font-semibold">
                            {comp.timings.sparse_search_ms.toFixed(2)} ms
                          </span>
                        </div>
                      )}

                      {comp.timings.rrf_merge_ms !== undefined && (
                        <div className="flex justify-between text-slate-400">
                          <span>RRF Fusion:</span>
                          <span className="text-slate-200 font-semibold">
                            {comp.timings.rrf_merge_ms.toFixed(2)} ms
                          </span>
                        </div>
                      )}

                      {comp.timings.rerank_ms !== undefined && comp.timings.rerank_ms > 0 && (
                        <div className="flex justify-between text-amber-400/90 font-medium">
                          <span>Cross-Encoder Rerank:</span>
                          <span className="text-amber-300 font-semibold">
                            {comp.timings.rerank_ms.toFixed(2)} ms
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Retrieved Paper IDs */}
                  <div className="space-y-1.5">
                    <div className="text-[11px] font-semibold text-slate-300 flex items-center justify-between">
                      <span>Retrieved Paper IDs ({comp.retrieved_paper_ids.length}):</span>
                    </div>

                    {comp.retrieved_paper_ids.length > 0 ? (
                      <div className="flex flex-wrap gap-1.5">
                        {comp.retrieved_paper_ids.map((pid, pidx) => (
                          <a
                            key={pidx}
                            href={`https://arxiv.org/abs/${pid}`}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-indigo-300 border border-slate-700 transition"
                            title={`View paper ${pid} on arXiv`}
                          >
                            {pid}
                            <ExternalLink className="w-2.5 h-2.5 opacity-70" />
                          </a>
                        ))}
                      </div>
                    ) : (
                      <div className="text-[11px] text-slate-500 italic">
                        No paper IDs returned
                      </div>
                    )}
                  </div>

                  {/* Top-Chunk Preview Snippet */}
                  <div className="space-y-1.5 pt-2 border-t border-slate-800/80">
                    <div className="text-[11px] font-semibold text-slate-300 flex items-center gap-1">
                      <FileText className="w-3 h-3 text-slate-400" />
                      Top-Chunk Preview:
                    </div>
                    {comp.top_chunk_preview ? (
                      <blockquote className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80 text-[11px] text-slate-300 italic leading-relaxed line-clamp-4">
                        "{comp.top_chunk_preview}"
                      </blockquote>
                    ) : (
                      <div className="text-[11px] text-slate-500 italic p-2 bg-slate-950/40 rounded border border-slate-800/40">
                        No preview returned by backend
                      </div>
                    )}
                    <span className="text-[9px] text-slate-500 block">
                      *Top-chunk preview snippet returned by endpoint
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Empty State when no results and not loading */}
      {!results && !loading && !error && (
        <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-12 text-center space-y-3">
          <div className="p-3 w-12 h-12 mx-auto rounded-xl bg-slate-800/60 text-slate-400 flex items-center justify-center">
            <Layers className="w-6 h-6" />
          </div>
          <h3 className="text-sm font-bold text-slate-300">
            No Ablation Comparison Run Yet
          </h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto leading-relaxed">
            Select one or more chunking collections, choose your retrieval strategies,
            and click "Run Ablation Comparison" to evaluate latency and retrieved paper
            candidates side by side.
          </p>
        </div>
      )}
    </div>
  );
};

export default RetrievalLab;
