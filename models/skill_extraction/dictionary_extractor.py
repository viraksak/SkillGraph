"""Dictionary-based Skill Extractor and Multi-Tier Skill Normalizer.

Provides:
1. Shared taxonomy loading utilities for both the 230-skill CSV taxonomy and
   the 31-skill JSON benchmark taxonomy.
2. `SkillNormalizer` for deterministic surface-to-canonical normalization with
   explicit match methods (`exact_match`, `alias_match`,
   `context_verified_short_token`, `country_specific_broad_match`,
   `unresolved_canonical`) and research-integrity safeguards against forcing
   uncertain or region-specific legal mappings.
3. `DictionarySkillExtractor` using boundary-aware regular expressions,
   all-uppercase acronym case enforcement (fixing e.g. "Siem Reap" vs "SIEM"),
   symbol protection (C++, C#, .NET, Node.js, CI/CD), local context
   disambiguation (C, R, Go), section tagging, and deterministic tie-breaking.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.preprocess_data import TextPreprocessor  # noqa: E402

# Short or polysemous tokens requiring strict casing and/or context checks
AMBIGUOUS_SHORT_TOKENS: frozenset[str] = frozenset({
    "c",
    "r",
    "go",
    "py",
    "js",
    "ts",
    "cv",
    "tf",
    "dl",
    "ml",
    "ai",
    "bi",
    "it",
    "qa",
    "ui",
    "ux",
})

SINGLE_LETTER_OR_GO_LOWER: frozenset[str] = frozenset({"c", "r", "go"})

# Explicit Week 3 skill-mention modality labels
SKILL_MODALITIES: tuple[str, ...] = (
    "required",
    "preferred",
    "mentioned",
    "negated_requirement",
    "excluded_or_prohibited",
    "contextually_irrelevant",
)

# Conservative pre-check regex to detect whether text may contain negation or
# prohibition cues (ignores the harmless boilerplate phrase "no prior experience").
NEGATION_CUE_PRECHECK_REGEX = re.compile(
    r"(?:\b(?:not|never|neither|nor|without|avoid(?:s|ed|ing)?|unnecessary|unneeded|"
    r"prohibit(?:s|ed|ing)?|forbid(?:s|den)?|ban(?:s|ned|ning)?|"
    r"exclud(?:e|es|ed|ing)|eschew(?:s|ed|ing)?|discontinu(?:e|es|ed|ing)|"
    r"deprecated|irrelevant)\b|n['’]t\b|\bno\b(?!\s+prior\s+experience\b))",
    flags=re.IGNORECASE,
)

CLAUSE_SPLIT_REGEX = re.compile(
    r"(?<=[.!?])\s+|\s*;\s*|,\s*(?=(?:but|however|whereas|while|yet|though|although)\b)|\s+(?=(?:but|however|whereas)\b)",
    flags=re.IGNORECASE,
)

_NEG_REQUIREMENT_VERBS = frozenset({
    "require", "requires", "required", "need", "needs", "needed",
    "expect", "expects", "expected", "mandatory", "essential", "necessary",
})
_UNNECESSARY_CUES = frozenset({"unnecessary", "unneeded", "irrelevant"})
_PROHIBITION_CUES = frozenset({
    "avoid", "avoids", "avoided", "avoiding",
    "prohibit", "prohibits", "prohibited", "prohibiting",
    "forbid", "forbids", "forbidden", "forbidding",
    "ban", "bans", "banned", "banning",
    "exclude", "excludes", "excluded", "excluding",
    "eschew", "discontinue", "discontinued",
})
_EXCLUSION_ACTION_VERBS = frozenset({
    "use", "using", "develop", "developing", "write", "writing",
    "build", "building", "deploy", "deploying", "code", "coding",
    "work", "working", "apply", "applying", "adopt", "adopting",
})
_POSITIVE_REQUIRED_CUES = re.compile(
    r"\b(?:must|require[sd]?|mandatory|essential|need(?:s|ed)?|minimum\s+requirements|"
    r"candidates\s+must|expect(?:s|ed)?|strong\s+grounding)\b",
    flags=re.IGNORECASE,
)
_POSITIVE_PREFERRED_CUES = re.compile(
    r"\b(?:prefer(?:red|s)?|advantage|desirable|bonus|nice-to-have|ideal\s+candidates|plus|optional)\b",
    flags=re.IGNORECASE,
)
_NON_TAXONOMY_PROHIBIT_REGEX = re.compile(
    r"\b(?P<verb>avoid(?:ing)?|prohibit(?:ing)?|forbid(?:ding)?|exclude|ban(?:ning)?)\s+"
    r"(?:using\s+)?(?P<target>(?:deprecated|legacy|unsupported|unapproved|proprietary)\s+[A-Za-z0-9_+.#/\- ]{3,50})",
    flags=re.IGNORECASE,
)

# Negative local contexts that indicate non-programming usage
GO_NEGATIVE_PHRASES: tuple[str, ...] = (
    "ready to go",
    "good to go",
    "way to go",
    "on the go",
    "let go",
    "to go to",
    "to go through",
    "to go beyond",
    "to go from",
    "go the extra",
    "go from ",
    "go to ",
    "go beyond",
    "go above",
    "go through",
    "go into ",
    "go out ",
    "go with ",
    "will go ",
    "can go ",
    "must go ",
    "would go ",
)

C_NEGATIVE_PHRASES: tuple[str, ...] = (
    "vitamin c",
    "series c",
    "section c",
    "appendix c",
    "part c",
    "schedule c",
    "plan c",
    "option c",
    "grade c",
    "c-suite",
    "c-level",
    "objective-c",
)

R_NEGATIVE_PHRASES: tuple[str, ...] = (
    "r&d",
    "r & d",
    "r-squared",
    "r/w",
    "vitamin r",
)

# Country-specific regulatory & accounting surface forms observed in the
# 2,000-posting synthetic corpus that map broadly to SG-SK-0048 or SG-SK-0058
# only when `include_broad_country_mappings=True`.
COUNTRY_SPECIFIC_BROAD_SURFACE_MAP: tuple[dict[str, Any], ...] = (
    # SG-SK-0048: Data Privacy Compliance
    {"phrase": "Vietnamese personal data protection rules", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "Australian Privacy Act compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "US state privacy laws compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "data protection requirements", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "Data Privacy Act compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "Privacy Act compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "DPDP Act compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "UAE PDPL compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "UK GDPR compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "PDP Law compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "PIPEDA compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "POPIA compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "PDPA compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "GDPR compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "LGPD compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "APPI compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    {"phrase": "PIPA compliance", "skill_id": "SG-SK-0048", "canonical_skill": "Data Privacy Compliance", "case_sensitive": False},
    # SG-SK-0058: Local Accounting Standards
    {"phrase": "Thai Financial Reporting Standards (TFRS)", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": False},
    {"phrase": "Australian Accounting Standards (AASB)", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": False},
    {"phrase": "Vietnamese Accounting Standards (VAS)", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": False},
    {"phrase": "Brazilian CPC standards", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": False},
    {"phrase": "UK GAAP (FRS 102)", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": False},
    {"phrase": "French PCG rules", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": False},
    {"phrase": "AASB standards", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": False},
    {"phrase": "IFRS and ASPE", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "Japanese GAAP", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": False},
    {"phrase": "Dutch GAAP", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": False},
    {"phrase": "NL GAAP", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "US GAAP", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "UK GAAP", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "J-GAAP", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "K-IFRS", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "SFRS(I)", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "Ind AS", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "CIFRS", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "IFRS", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "MFRS", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "PFRS", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "SFRS", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "TFRS", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "PSAK", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
    {"phrase": "HGB", "skill_id": "SG-SK-0058", "canonical_skill": "Local Accounting Standards", "case_sensitive": True},
)


def parse_aliases_field(raw_aliases: Any) -> list[str]:
    """Parses JSON-list, Python-list, or pipe-separated alias fields cleanly."""
    if raw_aliases is None:
        return []
    if isinstance(raw_aliases, (list, tuple, set)):
        return [str(a).strip() for a in raw_aliases if str(a).strip()]
    if pd.isna(raw_aliases):
        return []
    raw_str = str(raw_aliases).strip()
    if not raw_str:
        return []
    if raw_str.startswith("["):
        try:
            parsed = json.loads(raw_str)
            if isinstance(parsed, list):
                return [str(a).strip() for a in parsed if str(a).strip()]
        except Exception:
            pass
    if "|" in raw_str:
        return [part.strip() for part in raw_str.split("|") if part.strip()]
    return [raw_str]


def load_taxonomy_dataframe(taxonomy_source: pd.DataFrame | str | Path | dict[str, Any]) -> pd.DataFrame:
    """Loads CSV, JSON dict, or DataFrame taxonomies into a unified DataFrame."""
    if isinstance(taxonomy_source, pd.DataFrame):
        df = taxonomy_source.copy()
    elif isinstance(taxonomy_source, dict):
        rows = []
        for idx, (canonical, info) in enumerate(taxonomy_source.items(), start=1):
            rows.append({
                "skill_id": str(info.get("skill_id", f"JSON-SK-{idx:04d}")),
                "canonical_skill": str(canonical),
                "skill_category": str(info.get("category", info.get("skill_category", ""))),
                "skill_type": str(info.get("skill_type", "Technology")),
                "aliases": json.dumps(list(info.get("aliases", []))),
            })
        df = pd.DataFrame(rows)
    elif isinstance(taxonomy_source, (str, Path)):
        path = Path(taxonomy_source)
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return load_taxonomy_dataframe(data)
        df = pd.read_csv(path)
    else:
        raise TypeError(f"Unsupported taxonomy source type: {type(taxonomy_source)!r}")

    if "aliases" not in df.columns and "aliases_pipe_separated" in df.columns:
        df["aliases"] = df["aliases_pipe_separated"].apply(
            lambda v: json.dumps(parse_aliases_field(v))
        )
    if "aliases" not in df.columns:
        df["aliases"] = "[]"
    if "skill_category" not in df.columns:
        df["skill_category"] = ""
    if "skill_type" not in df.columns:
        df["skill_type"] = ""
    return df


def normalize_phrase_casing_rule(phrase: str, is_alias: bool) -> tuple[str, bool, str]:
    """Determines matching phrase text, strict-case requirement, and match method.

    Fixes Week 1 defect where only <= 3 uppercase tokens required strict case:
    1. Any all-uppercase acronym regardless of length (e.g. "SIEM", "GDPR",
       "IFRS", "SCADA", "HACCP", "SQL", "NLP", "R", "C") enforces strict case so
       "Siem" in "Siem Reap, Cambodia" never matches "SIEM".
    2. Lowercase JSON aliases ("c", "r", "go") are elevated to their unambiguous
       programming-language surface forms ("C", "R", "Go") with strict case.
    3. All tokens in AMBIGUOUS_SHORT_TOKENS (len <= 2/3) enforce strict case so
       common English words ("it", "ml", "go", "c", "r") do not false-fire.
    """
    stripped = phrase.strip()
    lower = stripped.lower()

    if lower == "c":
        return "C", True, "context_verified_short_token"
    if lower == "r":
        return "R", True, "context_verified_short_token"
    if lower == "go":
        return "Go", True, "context_verified_short_token"

    if stripped.isupper() or lower in AMBIGUOUS_SHORT_TOKENS:
        target = stripped.upper() if stripped.islower() and lower in AMBIGUOUS_SHORT_TOKENS else stripped
        # Special case: "Py" / "Go" handled above or kept exact
        return target, True, ("alias_match" if is_alias else "exact_match")

    return stripped, False, ("alias_match" if is_alias else "exact_match")


def is_valid_skill_context(
    text: str,
    start: int,
    end: int,
    matched_text: str,
    canonical_skill: str,
    allow_contextual_short_tokens: bool = True,
) -> bool:
    """Validates symbol boundaries and local lexical context around a match."""
    if not allow_contextual_short_tokens:
        return True

    canon_lower = canonical_skill.lower()
    matched_lower = matched_text.lower()

    # Protect single-letter C, R and short Go against symbol-adjacent and idiomatic hits
    is_c = matched_text == "C" or canon_lower == "c"
    is_r = matched_text == "R" or canon_lower in {"r", "r (programming language)"} and matched_text == "R"
    is_go = matched_text == "Go" or (canon_lower == "go" and matched_lower == "go")

    if is_c or is_r:
        if start > 0 and text[start - 1] in {"+", "#", "&", "/", "-", "."}:
            return False
        if end < len(text):
            next_ch = text[end]
            if next_ch in {"+", "#", "&", "/", "-"}:
                return False
            # Reject filename extensions like R.py / C.h, but allow sentence-final period "R."
            if next_ch == "." and end + 1 < len(text) and text[end + 1].isalnum():
                return False

    if is_c and matched_text == "C":
        window = text[max(0, start - 15):min(len(text), end + 15)].lower()
        for bad in C_NEGATIVE_PHRASES:
            if bad in window:
                return False

    if is_r and matched_text == "R":
        window = text[max(0, start - 15):min(len(text), end + 15)].lower()
        for bad in R_NEGATIVE_PHRASES:
            if bad in window:
                return False

    if is_go and matched_text == "Go":
        window = text[max(0, start - 20):min(len(text), end + 20)].lower()
        for bad in GO_NEGATIVE_PHRASES:
            if bad in window:
                return False

    return True


def _locate_clause_for_span(text: str, start: int, end: int) -> tuple[int, int, str]:
    """Splits text conservatively on sentence and contrastive clause boundaries."""
    boundaries = [0]
    for match in CLAUSE_SPLIT_REGEX.finditer(text):
        boundaries.append(match.end())
    boundaries.append(len(text))

    for i in range(len(boundaries) - 1):
        c_start = boundaries[i]
        c_end = boundaries[i + 1]
        if c_start <= start and end <= c_end:
            return c_start, c_end, text[c_start:c_end].strip()

    # Fallback narrow local window if span straddles a separator
    c_start = max(0, text.rfind(".", 0, start) + 1)
    nxt = text.find(".", end)
    c_end = nxt + 1 if nxt != -1 else len(text)
    return c_start, c_end, text[c_start:c_end].strip()


def infer_non_negated_modality(clause_text: str, source_section: str) -> str:
    """Assigns non-negated modality using section context and explicit clause cues."""
    if source_section == "required":
        return "required"
    if source_section == "preferred":
        return "preferred"
    if source_section == "intro_or_title":
        if _POSITIVE_REQUIRED_CUES.search(clause_text):
            return "required"
        if _POSITIVE_PREFERRED_CUES.search(clause_text):
            return "preferred"
        return "contextually_irrelevant"
    if _POSITIVE_REQUIRED_CUES.search(clause_text):
        return "required"
    if _POSITIVE_PREFERRED_CUES.search(clause_text):
        return "preferred"
    return "mentioned"


def classify_span_modality_regex(
    text: str,
    start: int,
    end: int,
    source_section: str,
) -> dict[str, Any]:
    """Conservative clause-scoped modality and negation classifier (fallback/peer)."""
    c_start, c_end, clause_text = _locate_clause_for_span(text, start, end)
    clause_lower = clause_text.lower()

    # Strip benign boilerplate phrase "no prior experience" before cue scanning
    masked_lower = re.sub(r"\bno\s+prior\s+experience\b", "   ", clause_lower)
    neg_hits = re.findall(r"\b(?:not|never|neither|nor|without)\b|n['’]t\b|\bno\b", masked_lower)
    prohib_hits = [w for w in _PROHIBITION_CUES if re.search(rf"\b{re.escape(w)}\b", masked_lower)]
    unnec_hits = [w for w in _UNNECESSARY_CUES if re.search(rf"\b{re.escape(w)}\b", masked_lower)]
    req_hits = [w for w in _NEG_REQUIREMENT_VERBS if re.search(rf"\b{re.escape(w)}\b", masked_lower)]
    act_hits = [w for w in _EXCLUSION_ACTION_VERBS if re.search(rf"\b{re.escape(w)}\b", masked_lower)]

    is_negated = False
    modality = infer_non_negated_modality(clause_text, source_section)
    sub_type: str | None = None

    if prohib_hits:
        is_negated = True
        modality = "excluded_or_prohibited"
        sub_type = "prohibited"
    elif unnec_hits and not neg_hits:
        is_negated = True
        modality = "negated_requirement"
        sub_type = "negated_requirement"
    elif neg_hits:
        # Check relative position or clause structure safely without over-negating
        if act_hits and not req_hits:
            is_negated = True
            modality = "excluded_or_prohibited"
            sub_type = "excluded"
        elif req_hits:
            is_negated = True
            modality = "negated_requirement"
            sub_type = "negated_requirement"
        elif any(tok in {"without", "neither", "nor", "no", "not", "never"} for tok in neg_hits):
            # Conservative: only negate if cue precedes the span inside the same clause
            rel_start = max(0, start - c_start)
            preceding = masked_lower[:rel_start]
            if re.search(r"\b(?:not|never|neither|nor|without|no)\b|n['’]t\b", preceding):
                is_negated = True
                modality = "excluded_or_prohibited" if act_hits else "negated_requirement"
                sub_type = "excluded" if act_hits else "negated_requirement"

    return {
        "modality": modality,
        "is_negated": is_negated,
        "polarity_evidence": {
            "analysis_method": "clause_scoped_regex",
            "prohibition_type": sub_type,
            "governing_token": (prohib_hits or req_hits or act_hits or [None])[0],
            "governing_lemma": (prohib_hits or req_hits or act_hits or [None])[0],
            "dependency_path": [],
            "negation_cues": sorted(set(neg_hits + prohib_hits + unnec_hits)),
            "clause_text": clause_text,
        },
    }


def extract_non_taxonomy_prohibited_clauses_regex(
    text: str,
    extracted_spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detects general prohibited technology clauses lacking canonical taxonomy skills."""
    results: list[dict[str, Any]] = []
    for match in _NON_TAXONOMY_PROHIBIT_REGEX.finditer(text):
        m_start, m_end = match.span("target")
        overlaps_skill = any(
            max(m_start, s["character_start"]) < min(m_end, s["character_end"])
            for s in extracted_spans
        )
        if overlaps_skill:
            continue
        target_phrase = match.group("target").strip(" .,;:")
        results.append({
            "target_phrase": target_phrase,
            "character_start": m_start,
            "character_end": m_start + len(target_phrase),
            "governing_verb": match.group("verb").lower(),
            "modality": "excluded_or_prohibited",
            "is_negated": True,
            "skill_id": None,
            "canonical_skill": None,
            "analysis_method": "clause_scoped_regex",
            "note": "Prohibited or deprecated technology phrase with no canonical taxonomy skill match.",
        })
    return results


class SkillNormalizer:
    """Multi-tier taxonomic normalizer grounded in the active repository taxonomy."""

    # Safe orthographical/abbreviation expansions (applied ONLY when target exists in loaded taxonomy)
    ORTHOGRAPHIC_VARIANTS: dict[str, tuple[str, ...]] = {
        "CI/CD": ("ci/cd", "ci cd", "cicd"),
        "Node.js": ("node.js", "node js", "nodejs"),
        ".NET": (".net", "dotnet", "dot net"),
        "scikit-learn": ("scikit-learn", "scikit learn", "sklearn"),
        "Scikit-Learn": ("scikit-learn", "scikit learn", "sklearn"),
        "Power BI": ("power bi", "powerbi", "microsoft power bi", "ms power bi"),
        "PostgreSQL": ("postgresql", "postgres", "psql"),
        "Kubernetes": ("kubernetes", "k8s"),
        "Natural Language Processing": ("natural language processing", "nlp"),
        "Machine Learning": ("machine learning", "ml"),
        "Deep Learning": ("deep learning", "dl"),
        "SQL": ("sql", "structured query language"),
        "React": ("react", "reactjs", "react.js"),
        "NoSQL": ("nosql", "no-sql", "no sql"),
        "C++": ("c++", "cpp", "c/c++"),
        "C#": ("c#", "csharp", "c-sharp"),
    }

    def __init__(
        self,
        taxonomy_df_or_path: pd.DataFrame | str | Path | dict[str, Any],
        include_broad_country_mappings: bool = False,
    ) -> None:
        self.taxonomy_df = load_taxonomy_dataframe(taxonomy_df_or_path)
        self.include_broad_country_mappings = include_broad_country_mappings

        self.exact_canonical: dict[str, dict[str, Any]] = {}
        self.lower_canonical: dict[str, dict[str, Any]] = {}
        self.exact_aliases: dict[str, dict[str, Any]] = {}
        self.lower_aliases: dict[str, dict[str, Any]] = {}
        self.broad_country_lookup: dict[str, dict[str, Any]] = {}
        self.skill_by_id: dict[str, dict[str, Any]] = {}

        self._build_index()

    def _build_index(self) -> None:
        for _, row in self.taxonomy_df.iterrows():
            skill_id = str(row["skill_id"])
            canonical = str(row["canonical_skill"])
            category = str(row.get("skill_category", ""))
            skill_type = str(row.get("skill_type", ""))

            record = {
                "skill_id": skill_id,
                "canonical_skill": canonical,
                "skill_category": category,
                "skill_type": skill_type,
            }
            self.skill_by_id[skill_id] = record
            self.exact_canonical[canonical] = record

            # Allow lowercase lookup in SkillNormalizer EXCEPT for strict-case "SIEM" or single chars "C", "R"
            if canonical not in {"C", "R", "Go", "SIEM"}:
                self.lower_canonical[canonical.lower()] = record

            aliases = parse_aliases_field(row.get("aliases"))
            for alias in aliases:
                self.exact_aliases.setdefault(alias, record)
                if alias not in {"C", "R", "Go", "SIEM", "c", "r", "go"}:
                    self.lower_aliases.setdefault(alias.lower(), record)

            # Register safe orthographic variants for skills actually present in this taxonomy
            for variant in self.ORTHOGRAPHIC_VARIANTS.get(canonical, ()):
                self.lower_aliases.setdefault(variant.lower(), record)

        # Build broad country-specific lookup ONLY for skill_ids present in loaded taxonomy
        for entry in COUNTRY_SPECIFIC_BROAD_SURFACE_MAP:
            target = self.skill_by_id.get(entry["skill_id"])
            if target is not None:
                key = entry["phrase"] if entry["case_sensitive"] else entry["phrase"].lower()
                self.broad_country_lookup[key] = {
                    **target,
                    "surface": entry["phrase"],
                    "case_sensitive": entry["case_sensitive"],
                }

    def normalize(
        self,
        mention: Any,
        context_text: str | None = None,
        allow_broad_country_mapping: bool | None = None,
    ) -> dict[str, Any]:
        """Normalizes a candidate mention string to a canonical taxonomy entry."""
        if not isinstance(mention, str) or not mention.strip():
            return {
                "input_mention": "" if mention is None else str(mention),
                "normalized_key": "",
                "skill_id": None,
                "canonical_skill": None,
                "skill_category": None,
                "skill_type": None,
                "match_method": "unresolved_canonical",
                "mapping_status": "UNRESOLVED_CANONICAL",
                "mapping_confidence": 0.0,
                "is_alias": False,
                "notes": "Empty or missing skill mention.",
            }

        cleaned = re.sub(r"\s+", " ", mention.strip())
        use_broad = (
            self.include_broad_country_mappings
            if allow_broad_country_mapping is None
            else allow_broad_country_mapping
        )

        # 1. Exact canonical match (handles C, R, Go, SIEM strictly by exact casing)
        if cleaned in self.exact_canonical:
            rec = self.exact_canonical[cleaned]
            short_tok = cleaned in {"C", "R", "Go"}
            if context_text and short_tok:
                standalone_matches = list(
                    re.finditer(rf"(?<![A-Za-z0-9]){re.escape(cleaned)}(?![A-Za-z0-9])", context_text)
                )
                if not standalone_matches:
                    return self._unresolved(cleaned, f"No standalone case-exact '{cleaned}' token in context_text.")
                valid_hit = any(
                    is_valid_skill_context(context_text, m.start(), m.end(), m.group(), rec["canonical_skill"])
                    for m in standalone_matches
                )
                if not valid_hit:
                    return self._unresolved(cleaned, "Rejected short token due to non-technical local context.")
            return {
                "input_mention": mention,
                "normalized_key": rec["canonical_skill"],
                **rec,
                "match_method": "context_verified_short_token" if short_tok else "exact_match",
                "mapping_status": "MAPPED",
                "mapping_confidence": 0.90 if short_tok else 1.0,
                "is_alias": False,
                "notes": None,
            }

        # 2. Exact alias match
        if cleaned in self.exact_aliases and cleaned not in {"c", "r", "go"}:
            rec = self.exact_aliases[cleaned]
            return {
                "input_mention": mention,
                "normalized_key": rec["canonical_skill"],
                **rec,
                "match_method": "alias_match",
                "mapping_status": "MAPPED",
                "mapping_confidence": 0.95,
                "is_alias": True,
                "notes": None,
            }

        # 3. Case-insensitive canonical match (excludes single chars C/R, Go, and SIEM Title-case "Siem")
        lowered = cleaned.lower()
        if lowered in self.lower_canonical:
            rec = self.lower_canonical[lowered]
            return {
                "input_mention": mention,
                "normalized_key": rec["canonical_skill"],
                **rec,
                "match_method": "exact_match",
                "mapping_status": "MAPPED",
                "mapping_confidence": 1.0,
                "is_alias": False,
                "notes": None,
            }

        # 4. Case-insensitive alias / safe abbreviation match (e.g., "nlp", "ml", "sql", "power bi")
        if lowered in self.lower_aliases:
            rec = self.lower_aliases[lowered]
            return {
                "input_mention": mention,
                "normalized_key": rec["canonical_skill"],
                **rec,
                "match_method": "alias_match",
                "mapping_status": "MAPPED",
                "mapping_confidence": 0.95,
                "is_alias": True,
                "notes": None,
            }

        # 5. Country-specific regulatory / accounting surface handling
        broad_hit = self.broad_country_lookup.get(cleaned) or self.broad_country_lookup.get(lowered)
        if broad_hit is not None:
            if broad_hit["case_sensitive"] and cleaned != broad_hit["surface"]:
                return self._unresolved(cleaned, "Case-sensitive regional accounting acronym mismatch.")
            if use_broad:
                return {
                    "input_mention": mention,
                    "normalized_key": broad_hit["canonical_skill"],
                    "skill_id": broad_hit["skill_id"],
                    "canonical_skill": broad_hit["canonical_skill"],
                    "skill_category": broad_hit["skill_category"],
                    "skill_type": broad_hit["skill_type"],
                    "match_method": "country_specific_broad_match",
                    "mapping_status": "BROAD_COUNTRY_MAPPING",
                    "mapping_confidence": 0.80,
                    "is_alias": True,
                    "notes": f"Broad regional mapping to {broad_hit['skill_id']} ({broad_hit['canonical_skill']}); distinct under strict annotation guidelines.",
                }
            return self._unresolved(
                cleaned,
                f"Country-specific regulation/accounting standard; distinct under strict guidelines unless broad mapping to {broad_hit['skill_id']} ({broad_hit['canonical_skill']}) is enabled.",
            )

        return self._unresolved(cleaned, "No verified canonical or alias entry in loaded taxonomy.")

    @staticmethod
    def _unresolved(cleaned: str, note: str) -> dict[str, Any]:
        return {
            "input_mention": cleaned,
            "normalized_key": cleaned,
            "skill_id": None,
            "canonical_skill": None,
            "skill_category": None,
            "skill_type": None,
            "match_method": "unresolved_canonical",
            "mapping_status": "UNRESOLVED_CANONICAL",
            "mapping_confidence": 0.0,
            "is_alias": False,
            "notes": note,
        }


class DictionarySkillExtractor:
    """Boundary-aware regular expression skill extractor with Week 2 & Week 3 enhancements."""

    def __init__(
        self,
        taxonomy_df_or_path: pd.DataFrame | str | Path | dict[str, Any],
        include_broad_country_mappings: bool = False,
        section_filter: Iterable[str] | None = None,
        enable_context_filters: bool = True,
        exclude_negated: bool = True,
    ) -> None:
        self.taxonomy_df = load_taxonomy_dataframe(taxonomy_df_or_path)
        self.include_broad_country_mappings = include_broad_country_mappings
        self.section_filter = set(section_filter) if section_filter is not None else None
        self.enable_context_filters = enable_context_filters
        self.exclude_negated = exclude_negated
        self.preprocessor = TextPreprocessor()
        self.normalizer = SkillNormalizer(
            self.taxonomy_df,
            include_broad_country_mappings=include_broad_country_mappings,
        )

        self.patterns: list[dict[str, Any]] = []
        self._build_patterns()

    @staticmethod
    def _compile_boundary_regex(phrase: str, case_sensitive: bool) -> re.Pattern[str]:
        """Compiles symbol- and case-aware boundary regex for technical & prose terms."""
        flags = 0 if case_sensitive else re.IGNORECASE
        escaped = re.escape(phrase)

        if phrase in {"C", "R"}:
            # Strict symbol isolation around C and R so C++, C#, C/C++, C-suite, R&D cannot fire
            pattern_str = rf"(?<![A-Za-z0-9+#&/\-\.]){escaped}(?![A-Za-z0-9+#&/\-])"
        else:
            pattern_str = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
            # Guard trailing '+' or '#' so e.g. 'C+' wouldn't partially match inside 'C++'
            if phrase.endswith("+") or phrase.endswith("#"):
                pattern_str = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9+#])"

        return re.compile(pattern_str, flags=flags)

    def _build_patterns(self) -> None:
        raw_entries: list[dict[str, Any]] = []
        seen_phrase_skill_pairs: set[tuple[str, str, bool]] = set()

        for _, row in self.taxonomy_df.iterrows():
            skill_id = str(row["skill_id"])
            canonical = str(row["canonical_skill"])
            category = str(row.get("skill_category", ""))
            skill_type = str(row.get("skill_type", ""))

            candidates: list[tuple[str, bool]] = [(canonical, False)]
            for alias in parse_aliases_field(row.get("aliases")):
                candidates.append((alias, True))

            for raw_phrase, is_alias in candidates:
                phrase, case_sensitive, match_method = normalize_phrase_casing_rule(raw_phrase, is_alias)
                dedup_key = (phrase if case_sensitive else phrase.lower(), skill_id, is_alias)
                if dedup_key in seen_phrase_skill_pairs:
                    continue
                seen_phrase_skill_pairs.add(dedup_key)

                is_short = match_method == "context_verified_short_token"
                confidence = 0.90 if is_short else (0.95 if is_alias else 1.0)
                raw_entries.append({
                    "phrase": phrase,
                    "case_sensitive": case_sensitive,
                    "skill_id": skill_id,
                    "canonical_skill": canonical,
                    "skill_category": category,
                    "skill_type": skill_type,
                    "is_alias": is_alias,
                    "match_method": match_method,
                    "mapping_status": "MAPPED",
                    "mapping_confidence": confidence,
                })

        if self.include_broad_country_mappings:
            for item in COUNTRY_SPECIFIC_BROAD_SURFACE_MAP:
                target = self.normalizer.skill_by_id.get(item["skill_id"])
                if target is not None:
                    raw_entries.append({
                        "phrase": item["phrase"],
                        "case_sensitive": item["case_sensitive"],
                        "skill_id": target["skill_id"],
                        "canonical_skill": target["canonical_skill"],
                        "skill_category": target["skill_category"],
                        "skill_type": target["skill_type"],
                        "is_alias": True,
                        "match_method": "country_specific_broad_match",
                        "mapping_status": "BROAD_COUNTRY_MAPPING",
                        "mapping_confidence": 0.80,
                    })

        raw_entries.sort(key=lambda x: (len(x["phrase"]), not x["is_alias"], x["mapping_confidence"]), reverse=True)

        for entry in raw_entries:
            self.patterns.append({
                **entry,
                "length": len(entry["phrase"]),
                "compiled": self._compile_boundary_regex(entry["phrase"], entry["case_sensitive"]),
            })

    def extract(self, text: Any, include_negated: bool = False) -> list[dict[str, Any]]:
        """Extracts normalized skill mentions with exact offsets and modality."""
        if not isinstance(text, str) or not text.strip():
            return []

        norm_text = self.preprocessor.normalize_preserving_offsets(text)["normalized_text"]
        sections = self.preprocessor.segment_job_sections(text)
        has_neg_cue = bool(NEGATION_CUE_PRECHECK_REGEX.search(norm_text))

        raw_matches: list[dict[str, Any]] = []
        for pat in self.patterns:
            for m in pat["compiled"].finditer(norm_text):
                start, end = m.span()
                matched_text = text[start:end]

                assert len(matched_text) == (end - start), (
                    f"Offset alignment error: expected length {end - start}, got {len(matched_text)}"
                )

                if self.enable_context_filters and not is_valid_skill_context(
                    norm_text, start, end, matched_text, pat["canonical_skill"]
                ):
                    continue

                source_section = self.preprocessor.get_section_for_span(text, start, end, sections)
                if self.section_filter is not None and source_section not in self.section_filter:
                    continue

                raw_matches.append({
                    "skill_mention": matched_text,
                    "matched_text": matched_text,
                    "canonical_skill": pat["canonical_skill"],
                    "skill_id": pat["skill_id"],
                    "skill_category": pat["skill_category"],
                    "skill_type": pat["skill_type"],
                    "character_start": start,
                    "character_end": end,
                    "match_length": end - start,
                    "is_alias": pat["is_alias"],
                    "extractor_name": "DictionarySkillExtractor",
                    "extraction_method": "dictionary_regex",
                    "match_method": pat["match_method"],
                    "mapping_status": pat["mapping_status"],
                    "mapping_confidence": pat["mapping_confidence"],
                    "source_section": source_section,
                })

        resolved = self._resolve_overlaps(raw_matches)
        output: list[dict[str, Any]] = []
        for sp in resolved:
            if not has_neg_cue and sp["source_section"] in {"required", "preferred"}:
                polarity = {
                    "modality": sp["source_section"],
                    "is_negated": False,
                    "polarity_evidence": {
                        "analysis_method": "section_context_fast_path",
                        "prohibition_type": None,
                        "governing_token": None,
                        "governing_lemma": None,
                        "dependency_path": [],
                        "negation_cues": [],
                        "clause_text": "",
                    },
                }
            else:
                polarity = classify_span_modality_regex(
                    norm_text,
                    sp["character_start"],
                    sp["character_end"],
                    sp["source_section"],
                )
            sp["modality"] = polarity["modality"]
            sp["is_negated"] = polarity["is_negated"]
            sp["polarity_evidence"] = polarity["polarity_evidence"]

            if self.exclude_negated and not include_negated and sp["is_negated"]:
                continue
            output.append(sp)

        return output

    def analyze_sentences_and_negation(self, text: Any) -> dict[str, Any]:
        """Returns positive, negated, and non-taxonomy prohibited clauses cleanly."""
        if not isinstance(text, str) or not text.strip():
            return {
                "positive_spans": [],
                "negated_spans": [],
                "all_spans": [],
                "non_taxonomy_prohibited_clauses": [],
                "analysis_method": "clause_scoped_regex",
            }
        all_spans = self.extract(text, include_negated=True)
        positive_spans = [s for s in all_spans if not s["is_negated"]]
        negated_spans = [s for s in all_spans if s["is_negated"]]
        non_tax = extract_non_taxonomy_prohibited_clauses_regex(text, all_spans)
        return {
            "positive_spans": positive_spans,
            "negated_spans": negated_spans,
            "all_spans": all_spans,
            "non_taxonomy_prohibited_clauses": non_tax,
            "analysis_method": "clause_scoped_regex",
        }

    @staticmethod
    def _resolve_overlaps(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Resolves overlapping matches deterministically.

        Priority order:
        1. Longest character span (`match_length`)
        2. Canonical exact name over alias (`not is_alias`)
        3. Higher `mapping_confidence`
        4. Earlier `character_start` offset
        """
        if not matches:
            return []

        sorted_matches = sorted(
            matches,
            key=lambda x: (
                x["match_length"],
                not x.get("is_alias", False),
                x.get("mapping_confidence", 1.0),
                -x["character_start"],
            ),
            reverse=True,
        )

        accepted_spans: list[dict[str, Any]] = []
        for candidate in sorted_matches:
            c_start = candidate["character_start"]
            c_end = candidate["character_end"]
            overlaps = any(
                max(c_start, a["character_start"]) < min(c_end, a["character_end"])
                for a in accepted_spans
            )
            if not overlaps:
                accepted_spans.append(candidate)

        accepted_spans.sort(key=lambda x: (x["character_start"], x["character_end"]))
        return accepted_spans
