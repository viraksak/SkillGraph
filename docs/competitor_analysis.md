# SkillGraph — Competitor Analysis

**Project:** SkillGraph: Explainable Skill Relationship Graph for University-to-Industry Skill Gap Analysis and Personalized Learning Path Recommendation  
**Prepared by:** Member 02 — Recommendation and Product Engineering Lead  
**Milestone:** 1 — Research and Planning  
**Document:** Competitor Analysis  
**Status:** Initial research draft

---

## 1. Introduction

### 1.1 Background

Students and recent graduates often face difficulties understanding which skills are required for their desired careers, identifying the differences between their current abilities and industry expectations, and deciding what they should learn next.

Several existing platforms provide career exploration, occupational information, skill classification, or career pathway recommendations. However, these platforms approach the problem from different perspectives.

This analysis reviews three relevant platforms:

1. O*NET / My Next Move
2. ESCO (European Skills, Competences, Qualifications and Occupations)
3. Lightcast Career Pathways

The purpose is to understand their features, identify useful design ideas, and determine how SkillGraph should position its proposed system.

### 1.2 Objectives

- Identify existing platforms related to career and skill analysis.
- Compare their main features and capabilities.
- Examine their relevance to SkillGraph.
- Identify opportunities for SkillGraph to provide additional value.
- Establish design considerations for the proposed system.

## 2. Competitor Overview

### 2.1 O*NET / My Next Move

**Website:** https://www.onetonline.org/  
**Career exploration tool:** https://www.mynextmove.org/

O*NET is an occupational information system that provides detailed information about occupations, including skills, knowledge, abilities, tasks, and work activities.

My Next Move is a simplified career exploration interface designed for job seekers and students. Users can explore careers through keyword searches, industry browsing, and interest-based career suggestions.

**Main features:**
- Occupational information and career descriptions.
- Skills and knowledge associated with occupations.
- Career exploration and keyword search.
- Interest-based career suggestions.
- Salary and occupational information.

**Strengths:**
- Provides structured occupational information.
- Supports exploration across a large number of occupations.
- Offers useful information for understanding career requirements.
- Provides a foundation for career guidance and occupational analysis.

**Limitations relevant to SkillGraph:**
- Its main purpose is occupational exploration and information rather than a complete student-specific learning pathway system.
- Students still need to interpret occupational requirements and decide how to address their individual skill gaps.
- Its standard career exploration experience is not the same as the proposed SkillGraph workflow, which combines student proficiency, skill relationships, and explainable learning recommendations.

**Lessons for SkillGraph:**

SkillGraph can adopt a clear occupation exploration interface and present occupational requirements in a structured, understandable format.

### 2.2 ESCO

**Website:** https://esco.ec.europa.eu/

ESCO is a European classification system for skills, competences, qualifications, and occupations. It provides a common terminology for connecting education and employment information.

ESCO organizes occupational and skill concepts and systematically represents relationships between them. Its data is available through an online portal and downloadable formats.

**Main features:**
- Standardized occupation classification.
- Structured skill and competence descriptions.
- Relationships between occupations and skills.
- Alternative labels and terminology.
- Multilingual occupational and skill information.
- Data access for other applications and systems.

**Strengths:**
- Provides a structured vocabulary for skills and occupations.
- Helps standardize different skill terms.
- Supports interoperability between employment and education systems.
- Offers a useful reference for skill taxonomy development.

**Limitations relevant to SkillGraph:**
- ESCO is primarily a classification framework, not a complete student-facing skill-gap and learning recommendation application.
- A standardized skill classification alone does not determine an individual student's proficiency.
- Personalized learning sequence generation is not the central purpose of the classification itself.

**Lessons for SkillGraph:**

SkillGraph should use a consistent skill dictionary and canonical skill names. Alternative labels and synonyms should be considered during skill normalization to reduce duplicate or inconsistent skill concepts.

### 2.3 Lightcast Career Pathways

**Website:** https://lightcast.io/  
**Documentation:** https://kb.lightcast.io/en/articles/6641056-lightcast-career-pathways

Lightcast Career Pathways is a labor-market intelligence solution that visualizes relationships between occupations and identifies possible career transitions.

Users can select a focus occupation and explore related roles, including advancement and lateral career transitions. The platform also identifies skills that may be important when moving from one occupation to another.

**Main features:**
- Visual representation of relationships between occupations.
- Career advancement and lateral transition pathways.
- Identification of skills important for occupational transitions.
- Skill-gap comparisons between occupations.
- Skill importance scores.

**Strengths:**
- Connects occupational relationships with skill requirements.
- Provides visual career pathway exploration.
- Helps identify skills that may support movement between occupations.
- Uses labor-market information to support career and workforce decisions.

**Limitations relevant to SkillGraph:**
- Its documented focus is occupational and workforce career pathways rather than the specific university-student workflow proposed by SkillGraph.
- A student may still need to provide and evaluate their own proficiency information to understand their personal learning needs.
- Its commercial data and models may not be directly reproducible within a student research project.

**Lessons for SkillGraph:**

SkillGraph should represent relationships between skills and occupations visually. It should also consider skill importance when prioritizing recommendations rather than treating every missing skill equally.

## 3. Competitor Comparison

| Evaluation dimension | O*NET / My Next Move | ESCO | Lightcast Career Pathways |
|---|---|---|---|
| Career exploration | Yes | Through occupation classification | Yes |
| Occupation information | Yes | Yes | Yes |
| Structured skill information | Yes | Yes | Yes |
| Skill normalization reference | Occupational data | Yes | Uses its own skill data |
| Occupation-skill relationships | Yes | Yes | Yes |
| Career pathway visualization | Career exploration | Not its primary purpose | Yes |
| Skill-gap analysis | Not its main student workflow | Not its primary purpose | Yes |
| Student-specific proficiency input | Not its main workflow | No | Not its primary documented focus |
| Personalized learning sequence | Not its main purpose | Not its main purpose | Career transition guidance, rather than the exact proposed student workflow |

*Note: This table compares documented primary capabilities and intended purposes. It does not claim that the platforms lack every capability not listed as a primary feature.*

## 4. Research Findings and Opportunities

The analysis reveals three important directions:

**1. Occupational information and career exploration**

O*NET demonstrates the value of structured occupational information and accessible career exploration.

**2. Standardized skill representation**

ESCO demonstrates the importance of using consistent skill terminology and connecting skills to occupations.

**3. Career pathways and skill importance**

Lightcast demonstrates the value of representing relationships between occupations and identifying important skills for career transitions.

These findings provide a foundation for SkillGraph's proposed design.

## 5. Proposed Positioning of SkillGraph

SkillGraph aims to combine relevant ideas from these systems into a student-oriented workflow.

The proposed workflow is:

1. Collect real English job descriptions for selected occupations.
2. Extract and normalize skills from the job descriptions.
3. Organize occupations and skills into a relationship graph.
4. Allow students to enter their current skill proficiency.
5. Compare student proficiency with occupation requirements.
6. Identify and prioritize individual skill gaps.
7. Generate an explainable learning pathway.

### Proposed target occupations

The initial project scope includes:

- Data Analyst
- Business Intelligence Analyst
- Data Scientist
- Data Engineer
- Machine Learning Engineer

### Potential differentiating direction

Rather than claiming that SkillGraph is the first system to provide career pathways or skill-gap analysis, the project will investigate whether combining the following elements provides useful value for university students:

- Job-description-based skill requirements.
- A structured skill relationship graph.
- Student-defined proficiency levels.
- Individual skill-gap prioritization.
- Prerequisite-aware learning pathways.
- Explanations for why each skill is recommended.

This is the project's proposed direction, not yet a proven research contribution. Its effectiveness must be evaluated through implementation and experiments.

## 6. Design Recommendations

Based on this initial competitor analysis, the following recommendations are proposed:

| Area | Recommendation |
|---|---|
| Career exploration | Provide a simple interface for selecting a target occupation. |
| Skill taxonomy | Maintain a consistent skill dictionary with canonical names and alternative labels. |
| Data source | Use real job descriptions as evidence of industry skill requirements. |
| Knowledge graph | Represent relationships between jobs, skills, prerequisites, and learning resources. |
| Student profile | Allow students to record their proficiency levels. |
| Gap analysis | Prioritize missing skills based on importance and current proficiency. |
| Learning recommendation | Consider prerequisites and explain why each skill should be learned. |
| Visualization | Provide an interactive graph that helps users understand skill relationships. |

## 7. Conclusion

The competitor analysis identifies three relevant foundations for SkillGraph: occupational exploration from O*NET, standardized skill representation from ESCO, and career pathway and skill-gap analysis from Lightcast.

These platforms provide useful references for designing the proposed system. SkillGraph will investigate a student-centered approach that connects job-description-based skill requirements, skill relationships, student proficiency, and explainable learning recommendations.

Further literature review and experimental evaluation are required before making claims about the system's novelty or effectiveness.

## 8. References

1. O*NET OnLine. https://www.onetonline.org/
2. My Next Move. https://www.mynextmove.org/
3. O*NET — My Next Move Help. https://www.onetonline.org/help/onet/mynextmove
4. European Commission. ESCO — About ESCO. https://esco.ec.europa.eu/en/about-esco
5. European Commission. ESCO — Frequently Asked Questions. https://esco.ec.europa.eu/en/about-esco/faq
6. Lightcast. Career Pathways. https://kb.lightcast.io/en/articles/6641056-lightcast-career-pathways
7. Lightcast. Career Pathways Data. https://docs.lightcast.io/data-shares/docs/career-pathways

---

**Document owner:** Member 02 — Recommendation and Product Engineering Lead  
**Project:** SkillGraph  
**Milestone:** 1 — Research and Planning  
**Document status:** Initial competitor analysis draft for team review
