import React, { useState, useRef } from "react";
import {
  Search,
  Sparkles,
  Clock,
  BookOpen,
  ExternalLink,
  Loader2,
  AlertCircle,
  FileText,
} from "lucide-react";
import { researchApi } from "../api/client";
import type { CollectionType, StrategyType, QueryResponse } from "../types/api";

export const ResearchAssistant: React.FC = () => {
  const [query, setQuery] = useState(
    "How does semantic chunking compare to fixed-size token windows for long-context retrieval?"
  );
  const [collection, setCollection] = useState<CollectionType>("arxiv_semantic_512");
  const [strategy, setStrategy] = useState<StrategyType>("hybrid_rerank");
  const [topK, setTopK] = useState<number>(5);
  const [generateAnswer, setGenerateAnswer] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
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
        query: query.trim(),
        collection,
        strategy,
        top_k: topK,
        generate_answer: generateAnswer,
      });
      setResult(res);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(
        typeof detail === "string"
          ? detail
          : "Failed to execute query against backend. Ensure the FastAPI service is running."
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
            className="mx-1 inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 hover:bg-indigo-500 hover:text-white transition cursor-pointer"
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
      {/* Search & Configuration Bar */}
      <form
        onSubmit={handleSearch}
        className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4 shadow-xl backdrop-blur"
      >
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask an academic research question over indexed arXiv papers..."
            className="w-full bg-slate-950 border border-slate-700/80 rounded-lg pl-11 pr-32 py-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 text-sm transition"
          />
          <Search className="w-5 h-5 text-slate-400 absolute left-3.5 top-4" />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="absolute right-2 top-2 bottom-2 px-5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-sm font-medium rounded-md flex items-center gap-2 transition cursor-pointer"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Sparkles className="w-4 h-4" />
            )}
            Run Query
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-1">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">
              Chunking Collection
            </label>
            <select
              value={collection}
              onChange={(e) => setCollection(e.target.value as CollectionType)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="arxiv_semantic_512">
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
            <label className="block text-xs font-medium text-slate-400 mb-1.5">
              Retrieval Strategy
            </label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as StrategyType)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="hybrid_rerank">
                Hybrid (BM25+RRF) + Cross-Encoder Rerank
              </option>
              <option value="hybrid">Hybrid (Dense + BM25 + RRF k=60)</option>
              <option value="dense">Dense Only (BGE-Small Cosine)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">
              Top-K Passages
            </label>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value={3}>Top-3 Chunks</option>
              <option value={5}>Top-5 Chunks</option>
              <option value={10}>Top-10 Chunks</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">
              Synthesis Mode
            </label>
            <div className="flex items-center h-[38px]">
              <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-300">
                <input
                  type="checkbox"
                  checked={generateAnswer}
                  onChange={(e) => setGenerateAnswer(e.target.checked)}
                  className="rounded border-slate-700 bg-slate-950 text-indigo-600 focus:ring-indigo-500 h-4 w-4"
                />
                Generate Answer (Groq LLM)
              </label>
            </div>
          </div>
        </div>
      </form>

      {/* Error State */}
      {error && (
        <div className="bg-red-950/40 border border-red-800 text-red-200 px-4 py-3 rounded-xl text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold block">Query Execution Error</span>
            <span className="text-xs text-red-300">{error}</span>
          </div>
        </div>
      )}

      {/* Empty Initial State */}
      {!result && !loading && !error && (
        <div className="text-center py-16 px-4 border border-dashed border-slate-800 rounded-2xl bg-slate-900/30">
          <FileText className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <h3 className="text-base font-semibold text-slate-300">
            No Active Query Result
          </h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto mt-1">
            Submit a question above to execute real-time retrieval across indexed
            arXiv research papers, view ranked source passages, and inspect stage latencies.
          </p>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && (
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-8 text-center space-y-3">
          <Loader2 className="w-8 h-8 text-indigo-400 animate-spin mx-auto" />
          <p className="text-sm font-medium text-slate-300">
            Retrieving & Reranking Passages...
          </p>
          <p className="text-xs text-slate-500 font-mono">
            {collection} | {strategy} | top_k={topK}
          </p>
        </div>
      )}

      {/* Results View */}
      {result && !loading && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Synthesis & Telemetry */}
          <div className="lg:col-span-7 space-y-4">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4 shadow-lg">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-indigo-400" />
                  <h3 className="font-semibold text-slate-100 text-sm">
                    Grounded Literature Synthesis
                  </h3>
                </div>
                <span className="text-xs font-mono text-slate-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                  {result.answer ? "Groq Llama-3.3-70B" : "Retrieval Only Mode"}
                </span>
              </div>

              <div className="prose prose-invert max-w-none text-slate-200 leading-relaxed whitespace-pre-line text-sm font-sans">
                {result.answer ? (
                  renderCitedAnswer(result.answer)
                ) : (
                  <span className="text-slate-400 italic">
                    Answer generation was bypassed. Retrieved context passages are
                    listed in the right-hand panel.
                  </span>
                )}
              </div>

              {/* Latency Telemetry Breakdown */}
              <div className="pt-4 border-t border-slate-800/80 flex flex-wrap gap-2 items-center text-xs font-mono">
                <span className="flex items-center gap-1 text-slate-400 mr-1">
                  <Clock className="w-3.5 h-3.5 text-indigo-400" /> Latency:
                </span>
                <span className="bg-slate-950 border border-slate-800 px-2 py-1 rounded text-slate-300">
                  Embed:{" "}
                  {(
                    result.timings.embedding_ms ??
                    result.timings.query_embedding_ms ??
                    0
                  ).toFixed(2)}
                  ms
                </span>
                <span className="bg-slate-950 border border-slate-800 px-2 py-1 rounded text-slate-300">
                  Dense:{" "}
                  {(
                    result.timings.dense_ms ??
                    result.timings.dense_search_ms ??
                    0
                  ).toFixed(2)}
                  ms
                </span>
                <span className="bg-slate-950 border border-slate-800 px-2 py-1 rounded text-slate-300">
                  BM25:{" "}
                  {(
                    result.timings.sparse_bm25_ms ??
                    result.timings.sparse_search_ms ??
                    0
                  ).toFixed(2)}
                  ms
                </span>
                {result.timings.rerank_ms !== undefined &&
                  result.timings.rerank_ms > 0 && (
                    <span className="bg-slate-950 border border-slate-800 px-2 py-1 rounded text-amber-300">
                      Rerank: {result.timings.rerank_ms.toFixed(2)}ms
                    </span>
                  )}
                {result.timings.generation_ms !== undefined &&
                  result.timings.generation_ms > 0 && (
                    <span className="bg-slate-950 border border-slate-800 px-2 py-1 rounded text-emerald-300">
                      LLM: {result.timings.generation_ms.toFixed(2)}ms
                    </span>
                  )}
                <span className="bg-indigo-950/60 border border-indigo-700/50 px-2 py-1 rounded text-indigo-300 font-semibold">
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

          {/* Right Column: Retrieved Passages */}
          <div className="lg:col-span-5 space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center justify-between">
              <span className="flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-indigo-400" />
                Retrieved Passages ({result.chunks.length})
              </span>
              <span className="font-mono text-slate-500 text-[11px]">
                Click passage or citation [X] to locate
              </span>
            </h3>

            <div className="space-y-3 max-h-[680px] overflow-y-auto pr-1">
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
                    className={`cursor-pointer rounded-xl p-4 border transition duration-200 ${
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
                            href={`https://arxiv.org/abs/${cleanArxivId}`}
                            target="_blank"
                            rel="noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="text-slate-400 hover:text-indigo-400 p-1"
                            title="Open paper on arXiv"
                          >
                            <ExternalLink className="w-3.5 h-3.5" />
                          </a>
                        )}
                      </div>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed font-sans line-clamp-6 hover:line-clamp-none transition-all">
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
