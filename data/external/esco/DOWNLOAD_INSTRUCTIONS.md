# ESCO (not downloaded)

Status: **empty.** The environment that built this dataset had no access to ESCO, so no ESCO file or identifier is included anywhere in `data/`.

What to download: the ESCO classification as CSV (English, latest version) from the ESCO portal, https://esco.ec.europa.eu/en/use-esco/download
Typical files (verify names after download): occupations, skills, occupation-skill relations, ISCO groups, skill hierarchy.
Place the unzipped CSVs in this folder. Keep the original file names and record the ESCO version number in a `VERSION.txt` file here.

Before use: read the ESCO terms of use on the portal and keep the required attribution. Licence terms were **not verified** by the builder.

After download: fill `esco_uri` in `processed/global_skill_taxonomy.csv` and `esco_occupation_uri` in `processed/global_occupation_mapping.csv`.
Only copy URIs that appear verbatim in the downloaded files. Never infer a URI from a label.
