import axios from "axios";
import type {
  QueryRequest,
  QueryResponse,
  CompareRequest,
  CompareResponse,
  BenchmarkSummaryResponse,
  HealthResponse,
  CollectionsResponse,
  AggregatedMetricRow,
  StatisticalTestRow,
  IndexStatRow,
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
    const { data } = await apiClient.get<{
      status: string;
      loaded_collections?: string[];
      groq_configured?: boolean;
    }>("/health");

    return {
      status: data.status || "healthy",
      loaded_collections: data.loaded_collections || [],
      groq_configured: Boolean(data.groq_configured),
    };
  },

  getCollections: async (): Promise<CollectionsResponse> => {
    const { data } = await apiClient.get<{
      available_collections?: string[];
      loaded_collections?: string[];
      index_statistics?: Record<string, unknown>[];
    }>("/collections");

    const rawStats = data.index_statistics || [];
    const normalizedStats: IndexStatRow[] = rawStats.map((item) => ({
      collection: String(item.collection || item.Collection || "Unknown"),
      vectors: Number(item.vectors ?? item.Vectors ?? 0),
      total_storage_mb: Number(
        item.total_storage_mb ?? item["Total Storage Footprint (MB)"] ?? 0
      ),
      storage_efficiency: Number(
        item.storage_efficiency ?? item["Storage Efficiency (MB per 1k)"] ?? 0
      ),
      total_build_time_s: Number(
        item.total_build_time_s ?? item["Total Build Time (s)"] ?? 0
      ),
      avg_embedding_ms: Number(
        item.avg_embedding_ms ?? item["Avg Embedding Time (ms/vec)"] ?? 0
      ),
      avg_insertion_ms: Number(
        item.avg_insertion_ms ?? item["Avg Insertion Time (ms/vec)"] ?? 0
      ),
      dimension: 384,
    }));

    return {
      available_collections: data.available_collections || [],
      loaded_collections: data.loaded_collections || [],
      index_statistics: normalizedStats,
    };
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
    const { data } = await apiClient.get<{
      total_benchmark_rows?: number;
      aggregated_metrics?: Record<string, unknown>[];
      statistical_significance?: Record<string, unknown>[];
    }>("/benchmarks/summary");

    const rawAgg = data.aggregated_metrics || [];
    const normalizedMetrics: AggregatedMetricRow[] = rawAgg.map((m) => ({
      collection: String(m.collection || m.Collection || ""),
      strategy: String(m.strategy || m.Strategy || ""),
      cutoff: Number(m.cutoff ?? m.Cutoff ?? m.top_k ?? m.TopK ?? 5),
      precision: Number(m.precision ?? m.Precision ?? 0),
      recall: Number(m.recall ?? m.Recall ?? 0),
      recall_10: m.recall_10 !== undefined || m.Recall_10 !== undefined
        ? Number(m.recall_10 ?? m.Recall_10)
        : undefined,
      mrr: Number(m.mrr ?? m.MRR ?? 0),
      ndcg: Number(m.ndcg ?? m.nDCG ?? m.ndcg_at_k ?? 0),
      citation_support_rate: Number(
        m.citation_support_rate ?? m.CitationSupportRate ?? 0
      ),
      unsupported_claim_rate: Number(
        m.unsupported_claim_rate ?? m.UnsupportedClaimRate ?? 0
      ),
      total_retrieval_ms: Number(
        m.total_retrieval_ms ?? m.TotalRetrievalMS ?? m.total_ms ?? 0
      ),
    }));

    const rawSig = data.statistical_significance || [];
    const normalizedSignificance: StatisticalTestRow[] = rawSig.map((s) => ({
      comparison_type: String(
        s.comparison_type || s["Comparison Type"] || "General"
      ),
      fixed_factor: String(s.fixed_factor || s["Fixed Factor"] || ""),
      group_a: String(s.group_a || s["Group A"] || ""),
      group_b: String(s.group_b || s["Group B"] || ""),
      metric: String(s.metric || s.Metric || "nDCG@5"),
      p_ttest: Number(s.p_ttest ?? 1),
      p_wilcoxon: Number(s.p_wilcoxon ?? 1),
      cohens_d: s.cohens_d !== undefined ? Number(s.cohens_d) : undefined,
      significant: Boolean(s.significant ?? s["Significant (p<0.05)"]),
    }));

    return {
      total_benchmark_rows: data.total_benchmark_rows || normalizedMetrics.length,
      aggregated_metrics: normalizedMetrics,
      statistical_significance: normalizedSignificance,
    };
  },
};
