import React, { useState, useEffect, Suspense, lazy } from "react";
import {
  BookOpen,
  BarChart3,
  Layers,
  Server,
  CheckCircle,
  ExternalLink,
  Loader2,
} from "lucide-react";
import { researchApi } from "./api/client";
import type { HealthResponse } from "./types/api";

const ResearchAssistant = lazy(() => import("./components/ResearchAssistant"));
const BenchmarkExplorer = lazy(() => import("./components/BenchmarkExplorer"));
const RetrievalLab = lazy(() => import("./components/RetrievalLab"));
const SystemOverview = lazy(() => import("./components/SystemOverview"));

type TabType = "assistant" | "benchmarks" | "lab" | "system";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>("assistant");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [checkingHealth, setCheckingHealth] = useState<boolean>(true);

  useEffect(() => {
    researchApi
      .getHealth()
      .then((res) => {
        setHealth(res);
      })
      .catch(() => {
        setHealth(null);
      })
      .finally(() => setCheckingHealth(false));
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="border-b border-slate-800/80 bg-slate-900/70 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-lg bg-gradient-to-tr from-indigo-600 to-indigo-400 flex items-center justify-center font-mono font-bold text-white shadow-md shadow-indigo-500/20">
              R
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold text-slate-100 tracking-tight">
                  ResearchGPT
                </h1>
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
                  RAG Ablation Lab
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                Systematic Benchmarking of Chunking Strategies & Hybrid Retrieval
              </p>
            </div>
          </div>

          {/* Backend Status Telemetry Pill */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs font-mono">
              <span className="flex h-2 w-2 relative">
                <span
                  className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                    health?.status === "healthy"
                      ? "bg-emerald-400"
                      : "bg-amber-400"
                  }`}
                />
                <span
                  className={`relative inline-flex rounded-full h-2 w-2 ${
                    health?.status === "healthy"
                      ? "bg-emerald-500"
                      : "bg-amber-500"
                  }`}
                />
              </span>
              <span className="text-slate-300">
                {checkingHealth
                  ? "Checking API..."
                  : health?.status === "healthy"
                  ? "FastAPI Online"
                  : "API Offline"}
              </span>
              {health?.groq_configured && (
                <span className="hidden md:inline-flex items-center gap-1 text-[11px] text-emerald-400 pl-1 border-l border-slate-700">
                  <CheckCircle className="w-3 h-3" /> Groq Active
                </span>
              )}
            </div>

            <a
              href="https://arxiv.org"
              target="_blank"
              rel="noreferrer"
              className="hidden lg:flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition"
            >
              arXiv Index <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </header>

      {/* Main Tab Navigation */}
      <div className="border-b border-slate-800/80 bg-slate-900/40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-2 sm:space-x-4">
            <button
              type="button"
              onClick={() => setActiveTab("assistant")}
              className={`flex items-center gap-2 py-3 px-3 text-xs sm:text-sm font-medium border-b-2 transition cursor-pointer ${
                activeTab === "assistant"
                  ? "border-indigo-500 text-indigo-400 font-semibold"
                  : "border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700"
              }`}
            >
              <BookOpen className="w-4 h-4" />
              Research Assistant
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("benchmarks")}
              className={`flex items-center gap-2 py-3 px-3 text-xs sm:text-sm font-medium border-b-2 transition cursor-pointer ${
                activeTab === "benchmarks"
                  ? "border-indigo-500 text-indigo-400 font-semibold"
                  : "border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700"
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              Benchmark Explorer
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("lab")}
              className={`flex items-center gap-2 py-3 px-3 text-xs sm:text-sm font-medium border-b-2 transition cursor-pointer ${
                activeTab === "lab"
                  ? "border-indigo-500 text-indigo-400 font-semibold"
                  : "border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700"
              }`}
            >
              <Layers className="w-4 h-4" />
              Retrieval Lab
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("system")}
              className={`flex items-center gap-2 py-3 px-3 text-xs sm:text-sm font-medium border-b-2 transition cursor-pointer ${
                activeTab === "system"
                  ? "border-indigo-500 text-indigo-400 font-semibold"
                  : "border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700"
              }`}
            >
              <Server className="w-4 h-4" />
              System Overview
            </button>
          </nav>
        </div>
      </div>

      {/* Main Content Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Suspense
          fallback={
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-12 text-center space-y-3 animate-pulse">
              <Loader2 className="w-8 h-8 animate-spin mx-auto text-indigo-400" />
              <div className="text-sm font-semibold text-slate-300">
                Loading view...
              </div>
              <p className="text-xs text-slate-500">
                Dynamic module loading in progress
              </p>
            </div>
          }
        >
          {activeTab === "assistant" && <ResearchAssistant />}
          {activeTab === "benchmarks" && <BenchmarkExplorer />}
          {activeTab === "lab" && <RetrievalLab />}
          {activeTab === "system" && <SystemOverview />}
        </Suspense>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 font-mono">
          <div>
            ResearchGPT: 24-Pipeline RAG Ablation Framework
          </div>
          <div className="flex flex-wrap items-center gap-4 text-[11px]">
            <span>Embedding: BAAI/bge-small-en-v1.5</span>
            <span>•</span>
            <span>Reranker: ms-marco-MiniLM-L-6-v2</span>
            <span>•</span>
            <span>ChromaDB: HNSW Cosine</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
