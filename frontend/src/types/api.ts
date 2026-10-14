export type CollectionType =
  | "arxiv_wholedoc"
  | "arxiv_fixed_512"
  | "arxiv_overlap_512_128"
  | "arxiv_semantic_512"
  | "arxiv_semantic";

export type StrategyType = "dense" | "hybrid" | "hybrid_rerank";

export interface LatencyBreakdown {
  embedding_ms?: number;
  query_embedding_ms?: number;
  dense_ms?: number;
  dense_search_ms?: number;
  sparse_bm25_ms?: number;
  sparse_search_ms?: number;
  rrf_fusion_ms?: number;
  rrf_merge_ms?: number;
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
  collection: string;
  strategy: string;
  timings: LatencyBreakdown;
  top_chunk_preview: string;
  retrieved_paper_ids: string[];
}

export interface CompareResponse {
  query: string;
  top_k?: number;
  comparisons: ComparisonCell[];
}

export interface AggregatedMetricRow {
  collection: string;
  strategy: string;
  cutoff: number;
  precision: number;
  recall: number;
  recall_10?: number;
  mrr: number;
  ndcg: number;
  citation_support_rate?: number;
  unsupported_claim_rate?: number;
  total_retrieval_ms?: number;
}

export interface StatisticalTestRow {
  comparison_type: string;
  fixed_factor: string;
  group_a: string;
  group_b: string;
  metric: string;
  p_ttest: number;
  p_wilcoxon: number;
  cohens_d?: number;
  significant: boolean;
}

export interface BenchmarkSummaryResponse {
  total_benchmark_rows: number;
  aggregated_metrics: AggregatedMetricRow[];
  statistical_significance: StatisticalTestRow[];
}

export interface HealthResponse {
  status: string;
  loaded_collections: string[];
  groq_configured: boolean;
}

export interface IndexStatRow {
  collection: string;
  vectors: number;
  total_storage_mb?: number;
  storage_efficiency?: number;
  total_build_time_s?: number;
  avg_embedding_ms?: number;
  avg_insertion_ms?: number;
  dimension?: number;
}

export interface CollectionsResponse {
  available_collections: string[];
  loaded_collections: string[];
  index_statistics: IndexStatRow[];
}
