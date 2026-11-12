# Annotation guidelines: skill mentions

You are labelling skill mentions in job-description text. Work from the text only. **Do not look at generator labels or model predictions**; independent labels are the point of this file.

## What to mark
A *skill mention* is a word or phrase that names something a candidate must know, be able to do, use or hold:
hard skills, soft skills, tools, technologies, knowledge areas, certifications, languages, competencies.

Mark the **shortest span that names the skill** ("Power BI", not "experience with Power BI"). Offsets are 0-based; `character_start` inclusive, `character_end` exclusive, into the exact `text` string. `text[start:end]` must equal `skill_mention`.

## What not to mark
- Duties and tasks ("prepare monthly bank reconciliations") unless they name a skill phrase on their own.
- Education levels, years of experience, job titles, employment type, locations, company or department names.
- Generic filler ("a strong team player" can be marked as `Teamwork` only if the text names the quality as a requirement; otherwise skip).
- Skills that are mentioned as something the employer does ("our team uses Python daily") unless the candidate is expected to have it.

## Fields
- `canonical_skill`: the name from `processed/global_skill_taxonomy.csv` if one clearly fits. If none fits, leave null and set status `UNRESOLVED_CANONICAL`. Do not map two different things to one skill because the names look alike. Different laws, standards or tools are different skills even if they play a similar role.
- `skill_category`: one of the allowed values in the template file. Use the category of the canonical skill; if unresolved, choose the closest and explain in `reviewer_notes`.
- `annotation_status`: `CONFIRMED` (sure), `UNCERTAIN` (a second reviewer should look), `UNRESOLVED_CANONICAL`, `REJECTED` (a proposed span is not a skill).
- `reviewer_notes`: write why for any non-obvious choice.

## Edge cases
- Lists ("Python, SQL and R"): one annotation per skill.
- Abbreviation and full name in the same text: annotate both occurrences separately.
- Required versus preferred is not part of this schema; the sentence context is kept in the text.
- Language names count as skills when the text says they are required or useful for the job.
- Nested phrases: mark the longest phrase that is one skill ("machine learning models" -> mark "machine learning").

## Quality process
1. At least two annotators label the same 10% of records. Compute agreement (span-level F1 or Cohen's kappa on categories) and report it.
2. Resolve disagreements by discussion and record the decision in `reviewer_notes`.
3. Only records with `record_status = COMPLETE` and no `UNCERTAIN` spans count as ground truth.
4. Keep annotation of REAL and SYNTHETIC records in separate evaluation sets.
