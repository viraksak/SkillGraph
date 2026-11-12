# SkillGraph global job-description dataset (data/)

> **Read this first.** This dataset contains **2,000 SYNTHETIC records and 0 REAL records.** Synthetic records are generated from templates and controlled vocabularies. They are **not** evidence of real labour-market demand, real skill frequencies or real country differences. Do not report any frequency, ranking or country comparison from synthetic data as a finding about the real world.

## 1. Purpose
Support SkillGraph experiments: skill extraction (dictionary, spaCy PhraseMatcher, ML), skill normalisation, occupation mapping, knowledge-graph construction and pipeline development. Synthetic data is for building and testing methods. Real data (none yet) is needed for any claim about the market.

## 2. Contents
| Path | What it is |
|---|---|
| `raw/synthetic_job_postings.csv/.json` | 2,000 synthetic records (same content in both files; list fields are JSON strings in the CSV, empty cell = null) |
| `raw/real_job_postings.csv/.json` | **Empty** (header only / `[]`). Reserved for real postings, kept separate from synthetic data |
| `processed/global_jobs_cleaned.csv/.json` | Validated copy of all records plus `description_char_len`, `dedup_cluster_id`, `split` |
| `processed/global_skill_mentions.csv` | 21,714 skill mentions with character offsets and canonical mapping (generator labels, **not human verified**) |
| `processed/global_skill_taxonomy.csv` | Local canonical skill list (230 skills, categories, types, aliases). Added file: needed to hold aliases and local IDs |
| `processed/global_occupation_mapping.csv` | 53 occupations with candidate ISCO-08 codes (unverified, see section 8). Added file name is as requested |
| `processed/data_quality_report.md` | Checks run and their results |
| `annotations/synthetic_job_postings_ground_truth.json` | Generator labels for every synthetic record. Status `GENERATOR_LABEL_UNREVIEWED`, `human_verified=false` |
| `annotations/human_annotation_template.json` | 300 blank records for human annotation (no pre-filled spans) |
| `annotations/annotation_guidelines.md` | Annotation rules |
| `external/esco`, `external/onet`, `external/isco` | Empty. Each holds `DOWNLOAD_INSTRUCTIONS.md` |

## 3. Coverage (synthetic)
Each of the 20 configured countries has 100 records (balanced on purpose for method testing; real data will not be balanced). All 15 required industries are present (IT is 6.65% of records). 53 occupations; the largest single occupation is 2.3% of records. Seniority: entry-level 309, junior 518, mid-level 558, senior 394, executive 221.

| Country | Records |
|---|---|
| Australia | 100 |
| Brazil | 100 |
| Cambodia | 100 |
| Canada | 100 |
| France | 100 |
| Germany | 100 |
| India | 100 |
| Indonesia | 100 |
| Japan | 100 |
| Malaysia | 100 |
| Netherlands | 100 |
| Philippines | 100 |
| Singapore | 100 |
| South Africa | 100 |
| South Korea | 100 |
| Thailand | 100 |
| United Arab Emirates | 100 |
| United Kingdom | 100 |
| United States | 100 |
| Vietnam | 100 |

| Industry | Records |
|---|---|
| Finance and Banking | 137 |
| Government and Public Administration | 136 |
| Tourism and Hospitality | 136 |
| Agriculture | 136 |
| Telecommunications | 134 |
| Sales and Marketing | 134 |
| Manufacturing | 134 |
| Energy and Utilities | 133 |
| Information Technology | 133 |
| Logistics and Supply Chain | 133 |
| Education | 132 |
| Healthcare | 131 |
| Retail and E-commerce | 131 |
| Media and Entertainment | 131 |
| Construction | 129 |

Regions used: Asia, Europe, North America, Oceania, Middle East, South America, Africa. Countries outside the 20 are **not covered**. Country codes are ISO 3166-1 alpha-2 and the country list is data, not code, so more countries can be added as rows with the same schema.

## 4. Schema
All records share the same fields in CSV and JSON.
`job_id, country, country_code, region, industry, occupation, job_title, company_name, location, description, responsibilities, required_skills, preferred_skills, education_level, experience_years_min, experience_years_max, employment_type, language_requirements, salary_min, salary_max, salary_currency, original_language, source_name, source_url, posted_at, collected_at, generated_at, data_type, is_synthetic, verified`
Added fields: `seniority_level, occupation_id, isco08_code_candidate, source_dataset, license, attribution, generator_seed`.
Cleaned file adds: `description_char_len, dedup_cluster_id, split`.
Unknown values are `null`. `required_skills` uses canonical names from `global_skill_taxonomy.csv` and includes required languages.

## 5. Real versus synthetic
- Synthetic: `data_type="SYNTHETIC"`, `is_synthetic=true`, `verified=false`, `source_name="AI_GENERATED"`, and null `company_name`, `source_url`, salary fields, `posted_at`, `collected_at`. All 2,000 records pass these checks.
- Real: **none imported.** No real dataset was downloaded because the build environment could not reach any dataset host. Nothing was invented to fill the gap.
- The two kinds must stay in separate files for experiments. `global_jobs_cleaned.*` can hold both later; always filter on `data_type`.

## 6. Generation methodology
Programmatic, no AI API. Each record combines: a country (cities, working languages, a local accounting-standard name and a data-protection law name used only as skill surface forms), an industry (organisation type, context sentences), an occupation (7 responsibility statements, required and preferred skill pools), and a seniority level (title, experience range, education adjustments, extra responsibilities). Wording is varied by randomly chosen sentence frames and alias surface forms. Fixed seed 42, fixed generation timestamp, so regeneration gives byte-identical files (verified). Near-duplicates are blocked at generation time (5-word-shingle Jaccard < 0.55 within an occupation) and re-checked afterwards: maximum observed 0.34, maximum TF-IDF cosine over all pairs 0.57, no pair at or above 0.80.
Descriptions are English. `original_language` is `en` for all records. Nothing is translated and presented as employer text.

## 7. Skill taxonomy
Local taxonomy `SKILLGRAPH_LOCAL_V1` with IDs `SG-SK-####`. These are **local IDs, not ESCO or O*NET IDs.** Skills carry one of the 18 requested categories and one of 8 types (hard skill, soft skill, tool, technology, knowledge area, certification, language, competency). `esco_uri` and `onet_element_id` are empty until the external files are downloaded. Mapping fields on each mention: `mapping_confidence`, `mapping_status`. Status values: `exact_match`, `alias_match`, `country_specific_surface_broad_mapping` (e.g. GDPR, PDPA, LGPD all map to the broad concept "Data Privacy Compliance"; these laws are not equivalent, so confidence is 0.8 and the status says so; same for national accounting standards). Mappings come from the generator, not from independent matching, so they cannot be used to evaluate a matcher without human review. Some category choices are judgement calls (e.g. Git and CI/CD are filed under Cloud computing).

## 8. Occupation taxonomy
53 occupations, each with a local ID `SG-OC-###` and a **candidate** ISCO-08 code and label written from memory, status `candidate_unverified_check_against_official_ilo_file`. Several are weak fits (e.g. Data Analyst, Machine Learning Engineer, Energy Analyst; confidence 0.4 to 0.6) because ISCO-08 has no exact unit group. Verify every code against the official ILO file before use. ESCO and O*NET codes are blank; no national classifications are included.

## 9. Data sources and licensing
No third-party data is included. Synthetic data was created for this project. Its licence is left unset (`license=null`); choose one before sharing. Candidate real sources to evaluate (**none downloaded, none verified, licences and coverage unchecked**): ESCO and O*NET and ISCO-08 (taxonomies, see `external/`); Hugging Face Hub datasets of job descriptions; Kaggle job-posting datasets; the EMSCAD employment-scam dataset from the University of the Aegean. For each, check the licence and terms on the dataset page, confirm that the data was not scraped against a site's terms, record real country coverage (many are mostly US/English), and keep the attribution text. Do not scrape sites.

## 10. Human annotation
`human_annotation_template.json` has 300 records (one per country and industry pair) with empty `annotations`. Spans are 0-based character offsets into the given `text`. Follow `annotation_guidelines.md`. Annotators should not see generator labels. Nothing here is human reviewed yet; annotation quality (agreement) is unmeasured. Use at least two annotators on a subset to measure agreement.

## 11. Splits
`split` (train 1594, validation 218, test 188) is assigned by a hash of `dedup_cluster_id`, so near-duplicate clusters stay together. At the 0.80 cosine threshold every record is its own cluster, which reflects the low similarity between records. For cross-country generalisation, hold out whole countries using `country_code`. Re-check clusters when real data is added.

## 12. Limitations and biases
- Synthetic frequencies reflect the generator's vocabularies, not the market. Co-occurrence and country differences are artefacts of design.
- Country balance is artificial. Country context is shallow: cities, languages, an accounting-standard name and a privacy-law name. Local job-market norms, titles and regulations are not modelled.
- Skill vocabulary is hand-written, English-only and tilted to the authors' choices. Occupation list is 53 roles.
- Every description follows a small number of sentence frames, so models may learn the templates. Do not use synthetic text alone to measure extraction performance.
- No salary, employer, date or demand information exists. No language detection was needed because all text is English by construction.
- Sample location lists are not representative of each country's geography.

## 13. How to add new job descriptions
1. Add real postings only from sources whose licence permits reuse. Save the original file under `external/` or keep it outside the repo, and note source, licence, access date.
2. Map fields to the schema in section 4 and write them to `raw/real_job_postings.csv/.json`. Set `data_type="REAL"`, `is_synthetic=false`, `verified=false`, fill `source_name`, `source_url`, `license`, `attribution`, `collected_at`. Keep `posted_at` and other unknown values `null`. Do not invent any value.
3. Use `job_id` like `GLOBAL_REAL_000001`. Never reuse `GLOBAL_SYN_` IDs.
4. Keep the original text unmodified. Offsets in annotations refer to the stored string.
5. Run the checks in `processed/data_quality_report.md` (unique IDs and descriptions, ISO codes, industry names, experience ranges, provenance, CSV/JSON equality) before adding to the cleaned files.
6. New countries or industries: add the ISO code and region exactly as in the existing rows; map industries to the 15 controlled names or document an extension.

## 14. Regeneration
The files were produced by a one-off script kept outside `data/` (you asked for no scripts in `data/`). Seed 42; rerunning it produced byte-identical files. If you want it in the repo, put it in `scripts/` and it will reproduce these files.
