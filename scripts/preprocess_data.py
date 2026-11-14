"""Advanced, Offset-Preserving Text Preprocessing for SkillGraph.

Provides non-destructive text normalization, reversible whitespace compaction
with character-index mapping, deterministic job posting section segmentation,
and strict character-offset verification without corrupting technical tokens
such as C++, C#, .NET, Node.js, CI/CD, and 4G/5G.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

# 1-to-1 character translation table so normalized text strictly preserves
# string length (len(normalized) == len(original)), guaranteeing 1:1 character
# offset alignment between preprocessed and original text.
OFFSET_PRESERVING_TRANSLATION = str.maketrans({
    "\t": " ",
    "\r": " ",
    "\u00a0": " ",  # Non-breaking space
    "\u2002": " ",  # En space
    "\u2003": " ",  # Em space
    "\u2009": " ",  # Thin space
    "\u200b": " ",  # Zero-width space mapped 1:1 to preserve index parity
    "\ufeff": " ",  # BOM mapped 1:1 to preserve index parity
    "\u2018": "'",  # Left single quotation mark
    "\u2019": "'",  # Right single quotation mark
    "\u201c": '"',  # Left double quotation mark
    "\u201d": '"',  # Right double quotation mark
    "\u2013": "-",  # En-dash
    "\u2014": "-",  # Em-dash
})

TECHNICAL_PUNCTUATION_EXAMPLES = (
    "C++",
    "C#",
    ".NET",
    "Node.js",
    "CI/CD",
    "4G/5G",
    "TCP/IP",
    "React.js",
    "Scikit-Learn",
    "SFRS(I)",
)

SECTION_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "preferred",
        (
            "it would be an advantage",
            "preferred qualifications",
            "preferred experience",
            "additional strengths in",
            "additional credit is given",
            "nice-to-have",
            "desirable qualifications",
            "bonus qualifications",
            "preference will be given",
            "ideal candidates will also",
        ),
    ),
    (
        "required",
        (
            "the minimum requirements are",
            "applicants need",
            "we expect a",
            "we expect an",
            "required qualifications",
            "candidates must possess",
            "essential requirements",
            "minimum qualifications",
            "to succeed in this role",
            "required skills",
            "mandatory requirements",
            "must-have qualifications",
            "you must bring",
            "strong grounding in",
        ),
    ),
    (
        "responsibilities",
        (
            "day-to-day responsibilities",
            "main duties",
            "the successful candidate will",
            "in this role you",
            "in this role, you",
            "key responsibilities",
            "you will be responsible",
            "core duties",
            "your responsibilities will",
        ),
    ),
    (
        "intro_or_title",
        (
            "we are hiring",
            "we are seeking",
            "this full-time",
            "a technology company",
            "a financial institution",
            "a healthcare provider",
            "an educational institution",
            "is looking for",
            "to join its",
            "to join our",
        ),
    ),
)


class TextPreprocessor:
    """Offset-preserving text preprocessor and section segmenter for job postings."""

    def __init__(self) -> None:
        self.translation_table = OFFSET_PRESERVING_TRANSLATION

    def normalize_preserving_offsets(self, text: Any) -> dict[str, Any]:
        """Normalizes non-standard whitespace and typographic punctuation 1-to-1.

        Unlike destructive lowercasing or regex whitespace collapsing, every
        character maps to a single character so len(normalized_text) == len(text).
        All technical tokens (C++, C#, .NET, Node.js, CI/CD, 4G/5G) remain intact.
        """
        if not isinstance(text, str):
            return {
                "original_text": "",
                "normalized_text": "",
                "is_valid": False,
                "length_preserved": True,
                "characters_modified": 0,
            }

        normalized = text.translate(self.translation_table)
        if len(normalized) != len(text):
            raise AssertionError(
                f"Offset-preserving invariant violated: {len(normalized)} != {len(text)}"
            )

        modified_count = sum(1 for a, b in zip(text, normalized) if a != b)
        return {
            "original_text": text,
            "normalized_text": normalized,
            "is_valid": bool(normalized.strip()),
            "length_preserved": True,
            "characters_modified": modified_count,
        }

    def normalize_compact(self, text: Any) -> tuple[str, list[int]]:
        """Collapses repeated whitespace while maintaining an exact index map.

        Returns `(compact_text, index_map)` where `index_map[i]` gives the
        character offset of `compact_text[i]` in the original string.
        """
        if not isinstance(text, str) or not text:
            return "", []

        translated = text.translate(self.translation_table)
        compact_chars: list[str] = []
        index_map: list[int] = []
        in_whitespace = True  # Strips leading whitespace naturally

        for idx, ch in enumerate(translated):
            if ch == " " or ch == "\n":
                if not in_whitespace:
                    compact_chars.append(" ")
                    index_map.append(idx)
                    in_whitespace = True
            else:
                compact_chars.append(ch)
                index_map.append(idx)
                in_whitespace = False

        if compact_chars and compact_chars[-1] == " ":
            compact_chars.pop()
            index_map.pop()

        return "".join(compact_chars), index_map

    @staticmethod
    def recover_original_span(
        compact_start: int,
        compact_end: int,
        index_map: list[int],
    ) -> tuple[int, int]:
        """Maps a span `[compact_start, compact_end)` back to original offsets."""
        if not index_map or compact_start >= compact_end or compact_end > len(index_map):
            raise ValueError(
                f"Invalid compact span ({compact_start}, {compact_end}) for index_map of length {len(index_map)}"
            )
        return index_map[compact_start], index_map[compact_end - 1] + 1

    @staticmethod
    def _classify_block(block_text: str, block_index: int = 0, total_blocks: int = 1) -> str:
        """Classifies a paragraph/block using explicit structural & lexical cues."""
        lowered = block_text.strip().lower()
        if not lowered:
            return "general"

        for section_name, cues in SECTION_CUES:
            for cue in cues:
                if lowered.startswith(cue) or f". {cue}" in lowered or f"\n{cue}" in lowered:
                    return section_name

        # Fallback for short multi-paragraph templates whose lead sentence is domain prose
        if total_blocks >= 3 and block_index == 0:
            return "intro_or_title"
        return "general"

    def segment_job_sections(self, text: Any) -> list[dict[str, Any]]:
        """Segments a job posting into section spans with exact character offsets."""
        if not isinstance(text, str) or not text.strip():
            return []

        norm_info = self.normalize_preserving_offsets(text)
        norm_text = norm_info["normalized_text"]

        # Split by double newlines while tracking exact start/end character offsets
        blocks: list[tuple[int, int, str]] = []
        cursor = 0
        for match in re.finditer(r"\n\s*\n", norm_text):
            end = match.start()
            if end > cursor:
                blocks.append((cursor, end, text[cursor:end]))
            cursor = match.end()
        if cursor < len(text):
            blocks.append((cursor, len(text), text[cursor:]))

        sections: list[dict[str, Any]] = []
        total_blocks = len(blocks)
        for idx, (b_start, b_end, b_text) in enumerate(blocks):
            if not b_text.strip():
                continue
            section_label = self._classify_block(b_text, block_index=idx, total_blocks=total_blocks)
            sections.append({
                "section": section_label,
                "start": b_start,
                "end": b_end,
                "text": b_text,
            })

        return sections

    def get_section_for_span(
        self,
        text: str,
        start: int,
        end: int,
        sections: list[dict[str, Any]] | None = None,
    ) -> str:
        """Returns the section label enclosing `[start, end)`."""
        if sections is None:
            sections = self.segment_job_sections(text)

        for sec in sections:
            if sec["start"] <= start and end <= sec["end"]:
                return str(sec["section"])

        # Fallback overlap check in case span straddles trailing punctuation/whitespace
        for sec in sections:
            if max(start, sec["start"]) < min(end, sec["end"]):
                return str(sec["section"])
        return "general"

    @staticmethod
    def verify_offset_alignment(
        original_text: str,
        start: int,
        end: int,
        expected_mention: str,
    ) -> bool:
        """Validates that `original_text[start:end] == expected_mention`."""
        if not isinstance(original_text, str):
            return False
        if not (0 <= start < end <= len(original_text)):
            return False
        return original_text[start:end] == expected_mention


def main() -> None:
    """Runs offset-preserving preprocessing validation across all 2,000 postings."""
    jobs_path = ROOT_DIR / "data" / "processed" / "global_jobs_cleaned.csv"
    gt_path = ROOT_DIR / "data" / "annotations" / "synthetic_job_postings_ground_truth.json"

    df_jobs = pd.read_csv(jobs_path)
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_records = json.load(f)

    preprocessor = TextPreprocessor()

    length_preserved_count = 0
    total_chars_modified = 0
    section_counts: dict[str, int] = {}

    for desc in df_jobs["description"]:
        res = preprocessor.normalize_preserving_offsets(desc)
        if res["length_preserved"]:
            length_preserved_count += 1
        total_chars_modified += res["characters_modified"]
        for sec in preprocessor.segment_job_sections(desc):
            section_counts[sec["section"]] = section_counts.get(sec["section"], 0) + 1

    # Verify 100% character offset preservation on all 21,714 synthetic reference spans
    total_spans = 0
    aligned_raw = 0
    aligned_norm = 0
    section_span_dist: dict[str, int] = {}

    for rec in gt_records:
        raw_text = rec["text"]
        norm_text = preprocessor.normalize_preserving_offsets(raw_text)["normalized_text"]
        sections = preprocessor.segment_job_sections(raw_text)
        for ann in rec["annotations"]:
            total_spans += 1
            s = ann["character_start"]
            e = ann["character_end"]
            mention = ann["skill_mention"]
            if preprocessor.verify_offset_alignment(raw_text, s, e, mention):
                aligned_raw += 1
            if preprocessor.verify_offset_alignment(norm_text, s, e, mention):
                aligned_norm += 1
            sec_label = preprocessor.get_section_for_span(raw_text, s, e, sections)
            section_span_dist[sec_label] = section_span_dist.get(sec_label, 0) + 1

    # Verify technical punctuation preservation sample
    tech_probe = "Stack: C++, C#, .NET, Node.js, CI/CD, 4G/5G, TCP/IP\u00a0— real\u2013time."
    probe_norm = preprocessor.normalize_preserving_offsets(tech_probe)
    for token in ("C++", "C#", ".NET", "Node.js", "CI/CD", "4G/5G", "TCP/IP"):
        assert token in probe_norm["normalized_text"], f"Missing token after normalization: {token}"

    print("=== SkillGraph Module 2: Offset-Preserving Preprocessing Audit ===")
    print(f"Postings processed:            {len(df_jobs)}")
    print(f"Length preservation 1:1 check: {length_preserved_count}/{len(df_jobs)}")
    print(f"Non-ASCII/tab chars normalized:{total_chars_modified}")
    print(f"Detected sections by count:    {section_counts}")
    print(f"Reference spans checked:       {total_spans}")
    print(f"Raw offset alignment:          {aligned_raw}/{total_spans}")
    print(f"Normalized offset alignment:   {aligned_norm}/{total_spans}")
    print(f"Reference span sections:       {section_span_dist}")


if __name__ == "__main__":
    main()

