import type { Metadata } from "next";
import React from "react";

export const metadata: Metadata = {
  title: "SkillGraph — Explainable Career Skill Gap & Learning Path Dashboard",
  description:
    "Knowledge Graph-Based Career Skill Gap Analysis, Priority Scoring, and Transparent Explanations.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
          backgroundColor: "#f8fafc",
          color: "#0f172a",
        }}
      >
        {children}
      </body>
    </html>
  );
}
