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
import {
  Server,
  Database,
  CheckCircle2,
  AlertCircle,
  HardDrive,
  Cpu,
  RefreshCw,
  Loader2,
  Activity,
  Layers,
  Clock,
} from "lucide-react";
import { researchApi } from "../api/client";
import type {
  HealthResponse,
  CollectionsResponse,
  IndexStatRow,
} from "../types/api";

const formatNumber = (num: number | undefined): string => {
  if (num === undefined || num === null) return "—";
  return num.toLocaleString();
};

const formatMetric = (
  val: number | undefined,
  decimals: number = 2,
  suffix: string = ""
): string => {
  if (val === undefined || val === null) return "—";
  return `${val.toFixed(decimals)}${suffix ? ` ${suffix}` : ""}`;
};

export const SystemOverview: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [collections, setCollections] = useState<CollectionsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSystemData = async (isManualRefresh = false) => {
    if (isManualRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);

    try {
      const [hData, cData] = await Promise.all([
        researchApi.getHealth(),
        researchApi.getCollections(),
      ]);
      setHealth(hData);
      setCollections(cData);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(
        typeof detail === "string"
          ? detail
          : "Failed to connect to backend endpoints (/health and /collections). Ensure the FastAPI server is running."
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchSystemData();
  }, []);

  if (loading) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-12 text-center space-y-3">
        <Loader2 className="w-8 h-8 animate-spin mx-auto text-indigo-400" />
        <div className="text-sm font-semibold text-slate-200">
          Fetching System Health & Index Telemetry...
        </div>
        <p className="text-xs text-slate-500">
          Querying GET /health and GET /collections
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-slate-900/60 border border-rose-500/30 rounded-2xl p-8 space-y-4">
        <div className="flex items-center gap-3 text-rose-400">
          <AlertCircle className="w-6 h-6" />
          <h3 className="text-sm font-bold">System Telemetry Unavailable</h3>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">{error}</p>
        <button
          type="button"
          onClick={() => fetchSystemData(false)}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Retry Connection
        </button>
      </div>
    );
  }

  const indexStats = collections?.index_statistics || [];
  const availableCols = collections?.available_collections || [];
  const loadedCols = collections?.loaded_collections || health?.loaded_collections || [];

  // Compute aggregate statistics only from present values
  const totalVectors = indexStats.reduce(
    (acc, curr) => acc + (curr.vectors || 0),
    0
  );

  const validStorageValues = indexStats
    .map((s) => s.total_storage_mb)
    .filter((v): v is number => v !== undefined && v !== null);
  const totalStorageMb = validStorageValues.length > 0
    ? validStorageValues.reduce((a, b) => a + b, 0)
    : undefined;

  const validBuildTimes = indexStats
    .map((s) => s.total_build_time_s)
    .filter((v): v is number => v !== undefined && v !== null);
  const totalBuildTimeS = validBuildTimes.length > 0
    ? validBuildTimes.reduce((a, b) => a + b, 0)
    : undefined;

  // Chart data
  const chartData = indexStats.map((row: IndexStatRow) => ({
    name: row.collection.replace("arxiv_", ""),
    fullName: row.collection,
    vectors: row.vectors,
    storage_mb: row.total_storage_mb ?? null,
    build_time_s: row.total_build_time_s ?? null,
  }));

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
              <Server className="w-5 h-5 text-emerald-400" />
              System Overview & Index Telemetry
            </h2>
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              GET /health · GET /collections
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1 max-w-3xl leading-relaxed">
            Real-time backend service diagnostics, active collection registry, vector
            store footprint, and indexing performance statistics returned by the FastAPI server.
          </p>
        </div>

        <button
          type="button"
          onClick={() => fetchSystemData(true)}
          disabled={refreshing}
          className="self-start sm:self-auto flex items-center gap-2 px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs text-slate-300 font-mono transition cursor-pointer disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? "animate-spin" : ""}`} />
          {refreshing ? "Refreshing..." : "Refresh Status"}
        </button>
      </div>

      {/* Top Telemetry KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Backend API Health */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400">FastAPI Service</span>
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-lg font-bold text-slate-100 font-mono capitalize">
              {health?.status || "Unknown"}
            </span>
          </div>
          <div className="text-[11px] text-slate-400 pt-1 border-t border-slate-800/80">
            HTTP Status 200 via Vite proxy
          </div>
        </div>

        {/* Loaded Collections in Memory */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400">In-Memory Cache</span>
            <Layers className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-lg font-bold text-indigo-400 font-mono">
            {loadedCols.length} / {availableCols.length}
          </div>
          <div className="text-[11px] text-slate-400 pt-1 border-t border-slate-800/80">
            Collections loaded in RETRIEVER_REGISTRY
          </div>
        </div>

        {/* Total Vectors Indexed */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400">Indexed Vectors</span>
            <Database className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-lg font-bold text-sky-400 font-mono">
            {formatNumber(totalVectors)}
          </div>
          <div className="text-[11px] text-slate-400 pt-1 border-t border-slate-800/80">
            Across {indexStats.length} chunking strategies
          </div>
        </div>

        {/* Groq Synthesis Configuration */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400">Groq LLM Status</span>
            <Cpu className="w-4 h-4 text-purple-400" />
          </div>
          <div className="flex items-center gap-1.5">
            {health?.groq_configured ? (
              <>
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-bold text-emerald-300 font-mono">
                  Configured
                </span>
              </>
            ) : (
              <>
                <AlertCircle className="w-4 h-4 text-amber-400" />
                <span className="text-sm font-bold text-amber-300 font-mono">
                  Unconfigured
                </span>
              </>
            )}
          </div>
          <div className="text-[11px] text-slate-400 pt-1 border-t border-slate-800/80 truncate">
            {health?.groq_configured
              ? "Synthesis API key active"
              : "Retrieval-only mode"}
          </div>
        </div>
      </div>

      {/* Available Collections Registry */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-bold text-slate-200">
              ChromaDB Collection Registry
            </h3>
          </div>
          <span className="text-[11px] font-mono text-slate-400">
            {availableCols.length} Registered Collections
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {availableCols.map((colName) => {
            const isLoaded = loadedCols.includes(colName);
            return (
              <div
                key={colName}
                className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-slate-200">
                    {colName}
                  </span>
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                      isLoaded
                        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                        : "bg-slate-800 text-slate-400 border-slate-700"
                    }`}
                  >
                    {isLoaded ? "Loaded (Hot)" : "On Disk"}
                  </span>
                </div>
                <div className="text-[11px] text-slate-400 leading-snug">
                  {colName === "arxiv_wholedoc" && "Whole document baseline without chunking"}
                  {colName === "arxiv_fixed_512" && "Fixed 512-token chunks, zero overlap"}
                  {colName === "arxiv_overlap_512_128" && "Sliding 512-token window with 128 overlap"}
                  {colName === "arxiv_semantic_512" && "Semantic boundary chunking (512 target)"}
                  {colName === "arxiv_semantic" && "Semantic chunking (legacy alias)"}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Index Statistics & Telemetry Table */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-sky-400" />
            <h3 className="text-sm font-bold text-slate-200">
              Storage Footprint & Ingestion Telemetry
            </h3>
          </div>
          <div className="text-[11px] font-mono text-slate-500">
            Source: <code className="text-slate-400">data/processed/index_statistics.csv</code>
          </div>
        </div>

        {indexStats.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-sans">
              <thead className="bg-slate-950/80 text-slate-400 uppercase font-mono text-[10px] border-b border-slate-800">
                <tr>
                  <th className="py-3 px-3">Collection</th>
                  <th className="py-3 px-3 text-right">Vectors</th>
                  <th className="py-3 px-3 text-right">Total Storage</th>
                  <th className="py-3 px-3 text-right">Efficiency (MB/1k)</th>
                  <th className="py-3 px-3 text-right">Build Time</th>
                  <th className="py-3 px-3 text-right">Avg Embedding</th>
                  <th className="py-3 px-3 text-right">Avg Insertion</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {indexStats.map((row: IndexStatRow, idx: number) => (
                  <tr
                    key={idx}
                    className="hover:bg-slate-800/30 transition-colors"
                  >
                    <td className="py-3 px-3 font-semibold text-slate-200">
                      {row.collection}
                    </td>
                    <td className="py-3 px-3 text-right text-indigo-400">
                      {formatNumber(row.vectors)}
                    </td>
                    <td className="py-3 px-3 text-right text-slate-300">
                      {formatMetric(row.total_storage_mb, 2, "MB")}
                    </td>
                    <td className="py-3 px-3 text-right text-slate-400">
                      {formatMetric(row.storage_efficiency, 2, "MB/1k")}
                    </td>
                    <td className="py-3 px-3 text-right text-slate-300">
                      {formatMetric(row.total_build_time_s, 2, "s")}
                    </td>
                    <td className="py-3 px-3 text-right text-slate-400">
                      {formatMetric(row.avg_embedding_ms, 2, "ms/vec")}
                    </td>
                    <td className="py-3 px-3 text-right text-slate-400">
                      {formatMetric(row.avg_insertion_ms, 2, "ms/vec")}
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot className="bg-slate-950/60 text-slate-300 font-mono text-[11px] border-t border-slate-800 font-semibold">
                <tr>
                  <td className="py-3 px-3">Total / Summary</td>
                  <td className="py-3 px-3 text-right text-indigo-400">
                    {formatNumber(totalVectors)}
                  </td>
                  <td className="py-3 px-3 text-right text-slate-300">
                    {formatMetric(totalStorageMb, 2, "MB")}
                  </td>
                  <td className="py-3 px-3 text-right text-slate-500">—</td>
                  <td className="py-3 px-3 text-right text-slate-300">
                    {formatMetric(totalBuildTimeS, 2, "s")}
                  </td>
                  <td className="py-3 px-3 text-right text-slate-500">—</td>
                  <td className="py-3 px-3 text-right text-slate-500">—</td>
                </tr>
              </tfoot>
            </table>
          </div>
        ) : (
          <div className="p-8 text-center bg-slate-950/50 rounded-xl border border-slate-800/80 text-xs text-slate-500">
            No index statistics returned by GET /collections.
          </div>
        )}
      </div>

      {/* Visual Chart: Vectors vs. Storage Footprint */}
      {chartData.length > 0 && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <HardDrive className="w-4 h-4 text-indigo-400" />
              <h3 className="text-sm font-bold text-slate-200">
                Chunking Strategy Storage Comparison
              </h3>
            </div>
            <span className="text-[11px] text-slate-500 font-mono">
              Vector Count vs. Storage Footprint (MB)
            </span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="name"
                  stroke="#64748b"
                  fontSize={12}
                />
                <YAxis
                  yAxisId="left"
                  orientation="left"
                  stroke="#6366f1"
                  fontSize={11}
                  tickFormatter={(val) => `${val}`}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#38bdf8"
                  fontSize={11}
                  tickFormatter={(val) => `${val}MB`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    borderColor: "#334155",
                    borderRadius: "0.75rem",
                    fontSize: "12px",
                  }}
                  formatter={(value: any, name: any) => [
                    name === "Storage Footprint (MB)" ? `${value} MB` : value,
                    name,
                  ]}
                />
                <Legend
                  verticalAlign="top"
                  height={36}
                  wrapperStyle={{ fontSize: "12px" }}
                />
                <Bar
                  yAxisId="left"
                  dataKey="vectors"
                  name="Total Vectors"
                  fill="#6366f1"
                  radius={[4, 4, 0, 0]}
                />
                <Bar
                  yAxisId="right"
                  dataKey="storage_mb"
                  name="Storage Footprint (MB)"
                  fill="#38bdf8"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Model Architecture & Index Architecture Notes */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
        <div className="flex items-center gap-2 text-slate-300 text-xs font-semibold">
          <Clock className="w-4 h-4 text-slate-400" />
          Technical Stack & Pipeline Configuration
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-400 leading-relaxed font-mono">
          <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-1">
            <div className="text-[11px] text-slate-300 font-semibold font-sans">
              Embedding Architecture
            </div>
            <div className="text-indigo-400 text-xs">BAAI/bge-small-en-v1.5</div>
            <p className="text-[10px] text-slate-500 font-sans">
              384-dimensional dense representations with normalized inner-product distance.
            </p>
          </div>

          <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-1">
            <div className="text-[11px] text-slate-300 font-semibold font-sans">
              Reranker Architecture
            </div>
            <div className="text-amber-400 text-xs">cross-encoder/ms-marco-MiniLM-L-6-v2</div>
            <p className="text-[10px] text-slate-500 font-sans">
              Full cross-attention scoring over concatenated query-passage token pairs.
            </p>
          </div>

          <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-1">
            <div className="text-[11px] text-slate-300 font-semibold font-sans">
              Vector Storage Engine
            </div>
            <div className="text-sky-400 text-xs">ChromaDB (HNSW Cosine Space)</div>
            <p className="text-[10px] text-slate-500 font-sans">
              Local persistent storage directory on disk, queried via parameter-driven retriever instances.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemOverview;
