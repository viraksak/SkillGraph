# O*NET (not downloaded)

Status: **empty.** No O*NET file or SOC/O*NET code is included anywhere in `data/`.

What to download: the O*NET database text files from the O*NET Resource Center, https://www.onetcenter.org/database.html
Useful files (verify names after download): occupation data, alternate titles, skills, technology skills, content model reference, and the O*NET-SOC to ISCO crosswalk if offered.
Place them in this folder and record the database release number in `VERSION.txt`.

Before use: check the licence and attribution statement on the O*NET page. Licence terms were **not verified** by the builder.

After download: fill `onet_soc_code` in `processed/global_occupation_mapping.csv` and `onet_element_id` in `processed/global_skill_taxonomy.csv` using codes copied from the files.
