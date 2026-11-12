# Data quality report

Dataset: 2,000 SYNTHETIC records, 0 REAL records. Checks ran on the files in `data/` after writing them. Result: **28 of 28 checks passed.**

## Checks
- CSV and JSON equal in content (raw synthetic and cleaned), 2,000 rows each.
- Real files are empty and separate from synthetic.
- Duplicate `job_id`: 0. Duplicate descriptions: 0. Empty descriptions: 0 (min length 847, median 1123, max 1419 characters).
- Country codes valid ISO 3166-1 alpha-2; country names match codes. Industry names all in the controlled list of 15.
- Experience ranges valid (min <= max, 0 <= min, max <= 40). Job ID format valid. Timestamp format valid.
- Synthetic provenance on every record (data_type, is_synthetic, verified, source_name, null company, URL, salary, dates).
- Required and preferred skills disjoint; no empty `required_skills`; list fields are lists.
- Near-duplicates: max within-occupation 5-gram Jaccard 0.34 (limit 0.55); max TF-IDF cosine over all pairs 0.57; no pair >= 0.80.
- Skill mentions: 21,714; every character span equals its mention text; mentions exactly cover required and preferred skills of each record; all canonical skills exist in the taxonomy; no external taxonomy IDs invented.
- Ground truth: no record marked human verified. Human template: 300 records, empty annotations, texts equal source records, one record per country and industry pair.
- Splits: cluster-consistent. Reproducibility: regenerating with seed 42 produced byte-identical files.

## Missing values (synthetic, by design)
company_name, salary_min, salary_max, salary_currency, source_url, posted_at, collected_at, license, attribution: 2,000 of 2,000 null. No other field has missing values.

## Distributions
Countries: 100 each. Industries: 129 to 137 each. Occupations: 53. IT share 6.65%. Education: Bachelor's degree 1114, Associate degree 310, Master's degree 277, Vocational certificate 202, High school diploma 80, Doctorate 17. Employment: Full-time 1725, Contract 159, Part-time 80, Internship 36.
Mapping status of mentions: alias_match 13875, exact_match 7688, country_specific_surface_broad_mapping 151.

## Not checked
Language detection (all text is English by construction), real-data checks (no real data), annotation agreement (no human annotations yet), correctness of candidate ISCO-08 codes (unverified), salary currency (no salaries).

## Missing coverage
Real records: none. Countries beyond the 20 listed: none. Non-English text: none. External taxonomies (ESCO, O*NET, ISCO-08 files): not downloaded.
