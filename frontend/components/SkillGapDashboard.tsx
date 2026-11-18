"use client";

import React, { useEffect, useState } from "react";
import {
  analyzeSkillGap,
  fetchOccupations,
  OccupationSummary,
  SkillGapAnalysisResponse,
} from "../lib/api";

export default function SkillGapDashboard() {
  const [occupations, setOccupations] = useState<OccupationSummary[]>([]);
  const [targetOccupation, setTargetOccupation] = useState("Data Analyst");
  const [studentId, setStudentId] = useState("SYNTHETIC_UI_STUDENT_01");
  const [skillsInput, setSkillsInput] = useState("Python:intermediate, SQL:beginner, Communication:advanced");
  const [excludeLanguage, setExcludeLanguage] = useState(true);
  const [result, setResult] = useState<SkillGapAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchOccupations()
      .then(setOccupations)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  async function handleAnalyze(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const parsedSkills = skillsInput
        .split(",")
        .map((chunk) => chunk.trim())
        .filter(Boolean)
        .map((chunk) => {
          const [name, level] = chunk.split(":").map((s) => s.trim());
          return { skill: name, proficiency_level: level || "intermediate" };
        });

      const data = await analyzeSkillGap({
        student_id: studentId.trim() || "SYNTHETIC_UI_STUDENT_01",
        target_occupation: targetOccupation,
        skills: parsedSkills,
        exclude_skill_types: excludeLanguage ? ["Language"] : [],
        top_k: 10,
      });
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 1040, margin: "0 auto", padding: "32px 20px" }}>
      <h1 style={{ marginBottom: 4 }}>SkillGraph — Explainable Skill Gap Analysis</h1>
      <p style={{ color: "#475569", marginTop: 0 }}>
        Synthetic 2,000-posting Knowledge Graph foundation | Transparent P(s) = alpha*G(s) + beta*O(s)
      </p>

      <form
        onSubmit={handleAnalyze}
        style={{
          background: "#ffffff",
          border: "1px solid #cbd5e1",
          borderRadius: 8,
          padding: 20,
          marginBottom: 24,
        }}
      >
        <div style={{ display: "grid", gap: 12, gridTemplateColumns: "1fr 1fr" }}>
          <label>
            <strong>Student ID</strong>
            <input
              value={studentId}
              onChange={(e) => setStudentId(e.target.value)}
              style={{ width: "100%", padding: 8, marginTop: 4 }}
            />
          </label>
          <label>
            <strong>Target Occupation</strong>
            <select
              value={targetOccupation}
              onChange={(e) => setTargetOccupation(e.target.value)}
              style={{ width: "100%", padding: 8, marginTop: 4 }}
            >
              {occupations.length === 0 ? (
                <option value="Data Analyst">Data Analyst</option>
              ) : (
                occupations.map((occ) => (
                  <option key={occ.occupation_id} value={occ.occupation}>
                    {occ.occupation} ({occ.industry})
                  </option>
                ))
              )}
            </select>
          </label>
        </div>

        <label style={{ display: "block", marginTop: 12 }}>
          <strong>Known Skills (format: Skill:proficiency, comma-separated)</strong>
          <input
            value={skillsInput}
            onChange={(e) => setSkillsInput(e.target.value)}
            style={{ width: "100%", padding: 8, marginTop: 4 }}
          />
        </label>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 14 }}>
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input
              type="checkbox"
              checked={excludeLanguage}
              onChange={(e) => setExcludeLanguage(e.target.checked)}
            />
            Exclude spoken Language skills from technical priority ranking
          </label>
          <button
            type="submit"
            disabled={loading}
            style={{
              background: "#0f172a",
              color: "#ffffff",
              padding: "10px 18px",
              borderRadius: 6,
              border: "none",
              cursor: "pointer",
            }}
          >
            {loading ? "Analyzing..." : "Run Skill Gap Analysis"}
          </button>
        </div>
      </form>

      {error && (
        <div style={{ background: "#fee2e2", color: "#991b1b", padding: 12, borderRadius: 6, marginBottom: 16 }}>
          {error}
        </div>
      )}

      {result && (
        <section>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: 12,
              marginBottom: 20,
            }}
          >
            <div style={{ background: "#fff", border: "1px solid #cbd5e1", borderRadius: 8, padding: 14 }}>
              <div style={{ fontSize: 12, color: "#64748b" }}>Required Coverage</div>
              <strong style={{ fontSize: 22 }}>
                {(result.coverage_metrics.required_skill_coverage * 100).toFixed(1)}%
              </strong>
            </div>
            <div style={{ background: "#fff", border: "1px solid #cbd5e1", borderRadius: 8, padding: 14 }}>
              <div style={{ fontSize: 12, color: "#64748b" }}>Weighted Coverage</div>
              <strong style={{ fontSize: 22 }}>
                {(result.coverage_metrics.overall_weighted_coverage * 100).toFixed(1)}%
              </strong>
            </div>
            <div style={{ background: "#fff", border: "1px solid #cbd5e1", borderRadius: 8, padding: 14 }}>
              <div style={{ fontSize: 12, color: "#64748b" }}>Skill Gap Score</div>
              <strong style={{ fontSize: 22 }}>{result.coverage_metrics.skill_gap_score.toFixed(4)}</strong>
            </div>
            <div style={{ background: "#fff", border: "1px solid #cbd5e1", borderRadius: 8, padding: 14 }}>
              <div style={{ fontSize: 12, color: "#64748b" }}>Prerequisite Graph</div>
              <strong style={{ fontSize: 13 }}>{result.learning_path.prerequisite_graph_status}</strong>
            </div>
          </div>

          <p style={{ fontSize: 13, color: "#475569", marginBottom: 16 }}>
            <strong>Limitation Notice:</strong> {result.learning_path.limitation_notice}
          </p>

          {result.priority_recommendations.map((rec) => (
            <article
              key={rec.skill_id}
              style={{
                background: "#fff",
                border: "1px solid #cbd5e1",
                borderRadius: 8,
                padding: 16,
                marginBottom: 12,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <strong>
                  #{rec.rank} {rec.canonical_skill} ({rec.skill_id}) — {rec.priority_tier}
                </strong>
                <span>P(s) = {rec.priority_score.toFixed(4)}</span>
              </div>
              <p style={{ margin: "6px 0", fontSize: 14 }}>{rec.explanation.why_identified}</p>
              <p style={{ margin: "4px 0", fontSize: 13, color: "#334155" }}>
                {rec.explanation.student_profile_status}
              </p>
              <p style={{ margin: "4px 0", fontSize: 13, color: "#334155" }}>
                {rec.explanation.multi_occupation_relevance}
              </p>
            </article>
          ))}
        </section>
      )}
    </main>
  );
}
