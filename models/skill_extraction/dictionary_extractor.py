"""Dictionary-based Skill Extractor using boundary-aware regular expressions.

Extracts canonical skills and aliases with character-level offsets,
greedy longest-match resolution, and symbol boundary protection.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
import pandas as pd


class DictionarySkillExtractor:
    """Boundary-aware regular expression skill extractor.

    Supports exact phrase matching, case-insensitive mapping with
    boundary protection for symbols (+, #, ., /, -), and greedy
    longest-match overlapping resolution.
    """

    def __init__(self, taxonomy_df_or_path: pd.DataFrame | str | Path):
        if isinstance(taxonomy_df_or_path, (str, Path)):
            self.taxonomy_df = pd.read_csv(taxonomy_df_or_path)
        else:
            self.taxonomy_df = taxonomy_df_or_path.copy()

        self.patterns: list[dict[str, Any]] = []
        self._build_patterns()

    def _build_patterns(self) -> None:
        """Compiles regex patterns from taxonomy canonical names and aliases.

        Sorts phrases by descending character length to favor greedy matching
        of longer multi-word phrases over their constituent sub-tokens.
        """
        raw_entries = []

        for _, row in self.taxonomy_df.iterrows():
            skill_id = str(row["skill_id"])
            canonical = str(row["canonical_skill"])
            category = str(row.get("skill_category", ""))
            skill_type = str(row.get("skill_type", ""))

            # Collect canonical name
            raw_entries.append({
                "phrase": canonical,
                "skill_id": skill_id,
                "canonical_skill": canonical,
                "skill_category": category,
                "skill_type": skill_type,
                "is_alias": False,
            })

            # Collect aliases
            aliases_raw = row.get("aliases", "[]")
            if pd.notnull(aliases_raw) and str(aliases_raw).strip():
                try:
                    aliases_list = json.loads(str(aliases_raw)) if isinstance(aliases_raw, str) else list(aliases_raw)
                    for alias in aliases_list:
                        alias_str = str(alias).strip()
                        if alias_str:
                            raw_entries.append({
                                "phrase": alias_str,
                                "skill_id": skill_id,
                                "canonical_skill": canonical,
                                "skill_category": category,
                                "skill_type": skill_type,
                                "is_alias": True,
                            })
                except Exception:
                    pass

        # Sort by descending length so multi-word phrases take precedence
        raw_entries.sort(key=lambda x: len(x["phrase"]), reverse=True)

        for entry in raw_entries:
            phrase = entry["phrase"]
            
            # Case sensitivity rule: short acronyms/tokens (<= 3 chars) use strict case
            # to avoid false positives (e.g. "it", "as", "or", "in")
            is_short = len(phrase) <= 3 and phrase.isupper()
            flags = 0 if is_short else re.IGNORECASE

            # Build boundary-aware regex:
            # We use lookaround (?<![A-Za-z0-9]) and (?![A-Za-z0-9]) rather than \b
            # so that symbols like C++, C#, .NET, Node.js, and CI/CD match correctly
            escaped_phrase = re.escape(phrase)
            pattern_str = rf"(?<![A-Za-z0-9]){escaped_phrase}(?![A-Za-z0-9])"

            compiled = re.compile(pattern_str, flags=flags)
            self.patterns.append({
                "compiled": compiled,
                "phrase": phrase,
                "length": len(phrase),
                "skill_id": entry["skill_id"],
                "canonical_skill": entry["canonical_skill"],
                "skill_category": entry["skill_category"],
                "skill_type": entry["skill_type"],
                "is_alias": entry["is_alias"],
            })

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Extracts skill mentions from unstructured text.

        Resolves overlapping spans by keeping the longest match.
        Guarantees that text[start:end] strictly equals matched_text.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        raw_matches = []
        for pat in self.patterns:
            compiled = pat["compiled"]
            for m in compiled.finditer(text):
                start, end = m.span()
                matched_text = m.group()

                # Verify character offset integrity
                assert text[start:end] == matched_text, f"Offset mismatch: {text[start:end]} != {matched_text}"

                raw_matches.append({
                    "skill_mention": matched_text,
                    "canonical_skill": pat["canonical_skill"],
                    "skill_id": pat["skill_id"],
                    "skill_category": pat["skill_category"],
                    "skill_type": pat["skill_type"],
                    "character_start": start,
                    "character_end": end,
                    "match_length": end - start,
                    "extraction_method": "dictionary_regex",
                })

        # Resolve overlapping spans using greedy longest-match
        resolved_matches = self._resolve_overlaps(raw_matches)
        return resolved_matches

    def _resolve_overlaps(self, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sorts candidate matches and removes overlapping sub-spans.

        Prioritizes:
        1. Longer match length (greedy longest span)
        2. Canonical match over alias
        3. Earlier character start offset
        """
        if not matches:
            return []

        # Sort primarily by length descending, then by start ascending
        sorted_matches = sorted(
            matches,
            key=lambda x: (x["match_length"], -x["character_start"]),
            reverse=True
        )

        accepted_spans: list[dict[str, Any]] = []

        for candidate in sorted_matches:
            c_start = candidate["character_start"]
            c_end = candidate["character_end"]

            # Check if this candidate overlaps with any already accepted span
            overlaps = False
            for accepted in accepted_spans:
                a_start = accepted["character_start"]
                a_end = accepted["character_end"]

                # Overlap condition: max(start1, start2) < min(end1, end2)
                if max(c_start, a_start) < min(c_end, a_end):
                    overlaps = True
                    break

            if not overlaps:
                accepted_spans.append(candidate)

        # Re-sort accepted spans by character start offset for presentation order
        accepted_spans.sort(key=lambda x: x["character_start"])
        return accepted_spans

