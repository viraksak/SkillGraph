"""spaCy PhraseMatcher based Skill Extractor.

Extracts technical skills using token-aware phrase matching with dual
case-sensitivity (LOWER for general terms, ORTH for sensitive acronyms).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher


class SpacySkillExtractor:
    """Token-aware phrase matching extractor using spaCy.

    Leverages spaCy's linguistic tokenization to guarantee exact token
    boundaries, preventing erroneous substring matches while preserving
    original character offsets.
    """

    def __init__(
        self,
        taxonomy_df_or_path: pd.DataFrame | str | Path,
        model_name: str = "en_core_web_sm"
    ):
        self.nlp = spacy.load(model_name)

        if isinstance(taxonomy_df_or_path, (str, Path)):
            self.taxonomy_df = pd.read_csv(taxonomy_df_or_path)
        else:
            self.taxonomy_df = taxonomy_df_or_path.copy()

        # Dual matcher architecture
        self.matcher_lower = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.matcher_exact = PhraseMatcher(self.nlp.vocab, attr="ORTH")
        self.pattern_metadata: dict[str, dict[str, Any]] = {}

        self._build_matchers()

    def _build_matchers(self) -> None:
        """Populates PhraseMatchers from canonical taxonomy names and aliases."""
        lower_patterns_by_id: dict[str, list[spacy.tokens.Doc]] = {}
        exact_patterns_by_id: dict[str, list[spacy.tokens.Doc]] = {}

        for _, row in self.taxonomy_df.iterrows():
            skill_id = str(row["skill_id"])
            canonical = str(row["canonical_skill"])
            category = str(row.get("skill_category", ""))
            skill_type = str(row.get("skill_type", ""))

            # Register metadata for this skill_id
            self.pattern_metadata[skill_id] = {
                "skill_id": skill_id,
                "canonical_skill": canonical,
                "skill_category": category,
                "skill_type": skill_type,
            }

            phrases_to_add = [canonical]

            aliases_raw = row.get("aliases", "[]")
            if pd.notnull(aliases_raw) and str(aliases_raw).strip():
                try:
                    aliases_list = json.loads(str(aliases_raw)) if isinstance(aliases_raw, str) else list(aliases_raw)
                    for alias in aliases_list:
                        alias_str = str(alias).strip()
                        if alias_str and alias_str not in phrases_to_add:
                            phrases_to_add.append(alias_str)
                except Exception:
                    pass

            for phrase in phrases_to_add:
                # Use exact case matcher for short acronyms/tokens (<= 3 chars, e.g. SQL, Git, CFA, PMP, JS)
                is_short = len(phrase) <= 3 and phrase.isupper()
                doc = self.nlp.make_doc(phrase)

                if is_short:
                    if skill_id not in exact_patterns_by_id:
                        exact_patterns_by_id[skill_id] = []
                    exact_patterns_by_id[skill_id].append(doc)
                else:
                    if skill_id not in lower_patterns_by_id:
                        lower_patterns_by_id[skill_id] = []
                    lower_patterns_by_id[skill_id].append(doc)

        for skill_id, docs in lower_patterns_by_id.items():
            self.matcher_lower.add(skill_id, docs)

        for skill_id, docs in exact_patterns_by_id.items():
            self.matcher_exact.add(skill_id, docs)

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Extracts skill entities from text, resolving overlaps greedily."""
        if not isinstance(text, str) or not text.strip():
            return []

        doc = self.nlp(text)
        raw_matches = []

        # 1. Evaluate lowercase matcher
        matches_lower = self.matcher_lower(doc)
        for match_id, start_idx, end_idx in matches_lower:
            span = doc[start_idx:end_idx]
            skill_id = self.nlp.vocab.strings[match_id]
            meta = self.pattern_metadata.get(skill_id, {})

            c_start = span.start_char
            c_end = span.end_char

            assert text[c_start:c_end] == span.text, f"Offset mismatch: {text[c_start:c_end]} != {span.text}"

            raw_matches.append({
                "skill_mention": span.text,
                "canonical_skill": meta.get("canonical_skill", ""),
                "skill_id": skill_id,
                "skill_category": meta.get("skill_category", ""),
                "skill_type": meta.get("skill_type", ""),
                "character_start": c_start,
                "character_end": c_end,
                "match_length": c_end - c_start,
                "extraction_method": "spacy_phrasematcher",
            })

        # 2. Evaluate exact orth matcher
        matches_exact = self.matcher_exact(doc)
        for match_id, start_idx, end_idx in matches_exact:
            span = doc[start_idx:end_idx]
            skill_id = self.nlp.vocab.strings[match_id]
            meta = self.pattern_metadata.get(skill_id, {})

            c_start = span.start_char
            c_end = span.end_char

            assert text[c_start:c_end] == span.text, f"Offset mismatch: {text[c_start:c_end]} != {span.text}"

            raw_matches.append({
                "skill_mention": span.text,
                "canonical_skill": meta.get("canonical_skill", ""),
                "skill_id": skill_id,
                "skill_category": meta.get("skill_category", ""),
                "skill_type": meta.get("skill_type", ""),
                "character_start": c_start,
                "character_end": c_end,
                "match_length": c_end - c_start,
                "extraction_method": "spacy_phrasematcher",
            })

        # Resolve overlaps greedily (longest match)
        return self._resolve_overlaps(raw_matches)

    def _resolve_overlaps(self, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not matches:
            return []

        # Sort descending by match length, then start offset
        sorted_matches = sorted(
            matches,
            key=lambda x: (x["match_length"], -x["character_start"]),
            reverse=True
        )

        accepted_spans: list[dict[str, Any]] = []

        for candidate in sorted_matches:
            c_start = candidate["character_start"]
            c_end = candidate["character_end"]

            overlaps = False
            for accepted in accepted_spans:
                a_start = accepted["character_start"]
                a_end = accepted["character_end"]
                if max(c_start, a_start) < min(c_end, a_end):
                    overlaps = True
                    break

            if not overlaps:
                accepted_spans.append(candidate)

        accepted_spans.sort(key=lambda x: x["character_start"])
        return accepted_spans

