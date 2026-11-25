export const API_BASE_URL =
  process.env.NEXT_PUBLIC_SKILLGRAPH_API_URL ?? "http://127.0.0.1:8000";

export interface OccupationSummary {
  occupation_id: string;
  occupation: string;
  industry: string;
  posting_count: number;
}

export interface PriorityCalculation {
  priority_score: number;
  formula: string;
  components: {
    G_gap_importance: number;
    O_external_opportunity: number;
    D_dependency_importance: number | null;
    D_status: string;
    C_learning_cost: number | null;
    C_status: string;
  };
}

export interface StructuredExplanation {
  skill: string;
  skill_id: string;
  why_identified: string;
  occupation_requirement_role: string;
  student_profile_status: string;
  importance_assessment: string;
  multi_occupation_relevance: string;
  priority_calculation: PriorityCalculation;
}

export interface SkillRecommendationItem {
  rank: number;
  skill_id: string;
  canonical_skill: string;
  skill_category: string;
  skill_type: string;
  requirement_type: string;
  profile_status: string;
  priority_score: number;
  priority_tier: string;
  learning_stage: string;
  explanation: StructuredExplanation;
}

export interface SkillGapAnalysisResponse {
  analysis_id: string;
  student_id: string;
  is_synthetic_profile: boolean;
  target_occupation: {
    occupation_id: string;
    occupation: string;
    industry: string;
    total_postings_analyzed: number;
  };
  coverage_metrics: {
    required_skill_coverage: number;
    overall_weighted_coverage: number;
    skill_gap_score: number;
    possessed_count: number;
    partially_demonstrated_count: number;
    missing_required_count: number;
    missing_preferred_count: number;
  };
  priority_recommendations: SkillRecommendationItem[];
  learning_path: {
    ordering_strategy: string;
    prerequisite_graph_status: string;
    learning_resources_status: string;
    limitation_notice: string;
  };
}

export async function fetchOccupations(search = ""): Promise<OccupationSummary[]> {
  const q = search ? `?search=${encodeURIComponent(search)}` : "";
  const res = await fetch(`${API_BASE_URL}/occupations${q}`);
  if (!res.ok) throw new Error(`Failed to load occupations (${res.status})`);
  return res.json();
}

export async function analyzeSkillGap(payload: {
  student_id: string;
  target_occupation: string;
  skills: Array<{ skill: string; proficiency_level: string }>;
  exclude_skill_types: string[];
  top_k: number;
}): Promise<SkillGapAnalysisResponse> {
  const res = await fetch(`${API_BASE_URL}/skill-gap/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, is_synthetic: true }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `Analysis failed (${res.status})`);
  }
  return res.json();
}
