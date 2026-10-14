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
  BarChart3,
  Award,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Loader2,
  Database,
} from "lucide-react";
import { researchApi } from "../api/client";
import type { AggregatedMetricRow, StatisticalTestRow } from "../types/api";

export const BenchmarkExplorer: React.FC = () => {
  const [metrics, setMetrics] = useState<AggregatedMetricRow[]>([]);
  const [stats, setStats] = useState<StatisticalTestRow[]>([]);
  const [cutoffFilter, setCutoffFilter] = useState<number>(5);
  const [totalRows, setTotalRows] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    researchApi
      .getBenchmarkSummary()
      .then((data) => {
        setMetrics(data.aggregated_metrics || []);
        setStats(data.statistical_significance || []);
        setTotalRows(data.total_benchmark_rows || 0);
      })
      .catch((err: any) => {
        const detail = err?.response?.data?.detail;
        setError(
          typeof detail === "string"
            ? detail
            : "Could not load benchmark summary from backend. Verify FastAPI service is running."
        );
      })
      .finally(() => setLoading(false));
  }, []);

  const filteredMetrics = metrics
    .filter((m) => Number(m.cutoff) === cutoffFilter)
    .map((m) => ({
      ...m,
      label: `${m.collection.replace("arxiv_", "")} (${m.strategy})`,
    }));

  if (loading) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-12 text-center space-y-3">
        <Loader2 className="w-8 h-8 text-indigo-400 animate-spin mx-auto" />
        <p className="text-sm font-medium text-slate-300">
          Loading Empirical Ablation Benchmark Data...
        </p>
        <p className="text-xs text-slate-500 font-mono">
          Querying /benchmarks/summary
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-950/40 border border-red-800 text-red-200 p-5 rounded-2xl flex items-start gap-3 text-sm">
        <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold block">Failed to Load Benchmark Telemetry</span>
          <span className="text-xs text-red-300">{error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Benchmark Header & Filter */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-indigo-500/10 border border-indigo-500/30 text-indigo-400">
            <BarChart3 className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-100">
              24-Configuration Ablation Benchmark Explorer
            </h2>
            <p className="text-xs text-slate-400 flex items-center gap-2 mt-0.5 font-mono">
              <Database className="w-3.5 h-3.5 text-slate-500" />
              Evaluated over {totalRows} test runs across 4 chunking strategies & 3 retrieval architectures
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 rounded-lg p-1.5 self-stretch sm:self-auto justify-end">
          <span className="text-xs text-slate-400 px-2 font-mono">Cutoff (K):</span>
          {[3, 5].map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => setCutoffFilter(k)}
              className={`px-3 py-1 rounded text-xs font-mono font-medium transition cursor-pointer ${
                cutoffFilter === k
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Top-{k}
            </button>
          ))}
        </div>
      </div>

      {/* Main Recharts Bar Chart */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">
              Retrieval Quality Metrics Across Configurations (K = {cutoffFilter})
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Empirical nDCG@{cutoffFilter}, Mean Reciprocal Rank (MRR), Precision@{cutoffFilter}, and Recall@{cutoffFilter}
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500">
            {filteredMetrics.length} configs displayed
          </span>
        </div>

        {filteredMetrics.length === 0 ? (
          <div className="text-center py-16 text-slate-500 text-sm">
            No metrics available for Cutoff K = {cutoffFilter}.
          </div>
        ) : (
          <div className="h-80 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={filteredMetrics}
                margin={{ top: 10, right: 10, left: -10, bottom: 50 }}
              >
                <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="label"
                  stroke="#94a3b8"
                  fontSize={11}
                  interval={0}
                  angle={-25}
                  textAnchor="end"
                  tick={{ fill: "#94a3b8" }}
                />
                <YAxis
                  stroke="#94a3b8"
                  fontSize={11}
                  domain={[0, 1]}
                  tick={{ fill: "#94a3b8" }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    borderColor: "#334155",
                    fontSize: "12px",
                    borderRadius: "8px",
                    boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.5)",
                  }}
                  itemStyle={{ padding: "2px 0" }}
                />
                <Legend
                  verticalAlign="top"
                  wrapperStyle={{ paddingBottom: "12px", fontSize: "12px" }}
                />
                <Bar dataKey="ndcg" fill="#6366f1" name={`nDCG@${cutoffFilter}`} radius={[3, 3, 0, 0]} />
                <Bar dataKey="mrr" fill="#10b981" name="MRR" radius={[3, 3, 0, 0]} />
                <Bar dataKey="precision" fill="#38bdf8" name={`Precision@${cutoffFilter}`} radius={[3, 3, 0, 0]} />
                <Bar dataKey="recall" fill="#f59e0b" name={`Recall@${cutoffFilter}`} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Paired Statistical Significance & Cohen's d Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-lg space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
          <div className="flex items-center gap-2">
            <Award className="w-5 h-5 text-emerald-400" />
            <div>
              <h3 className="text-sm font-semibold text-slate-100">
                Paired Statistical Hypothesis Testing & Effect Size
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Paired Student's t-test, Wilcoxon Signed-Rank test, and Cohen's d effect size on nDCG@5
              </p>
            </div>
          </div>
          <span className="text-xs font-mono text-slate-500">
            {stats.length} pairwise comparisons
          </span>
        </div>

        {stats.length === 0 ? (
          <div className="text-center py-10 text-slate-500 text-sm font-mono">
            No statistical significance tests recorded in backend.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2.5 px-3">Type</th>
                  <th className="py-2.5 px-3">Fixed Factor</th>
                  <th className="py-2.5 px-3">Pair (Group A vs Group B)</th>
                  <th className="py-2.5 px-3">Metric</th>
                  <th className="py-2.5 px-3">p (t-test)</th>
                  <th className="py-2.5 px-3">p (Wilcoxon)</th>
                  <th className="py-2.5 px-3">Cohen's d</th>
                  <th className="py-2.5 px-3">Significant (α=0.05)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {stats.map((row, i) => (
                  <tr key={i} className="hover:bg-slate-950/50 transition">
                    <td className="py-2.5 px-3 text-slate-400">
                      {row.comparison_type}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {row.fixed_factor}
                    </td>
                    <td className="py-2.5 px-3 text-indigo-300 font-semibold">
                      {row.group_a} <span className="text-slate-500">vs</span> {row.group_b}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {row.metric}
                    </td>
                    <td className="py-2.5 px-3 text-slate-200">
                      {Number(row.p_ttest).toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-200">
                      {Number(row.p_wilcoxon).toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-amber-300">
                      {row.cohens_d !== undefined ? Number(row.cohens_d).toFixed(3) : "—"}
                    </td>
                    <td className="py-2.5 px-3">
                      {row.significant ? (
                        <span className="inline-flex items-center gap-1 text-emerald-400 font-medium">
                          <CheckCircle2 className="w-3.5 h-3.5" /> Yes
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-slate-500">
                          <XCircle className="w-3.5 h-3.5" /> No
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default BenchmarkExplorer;
