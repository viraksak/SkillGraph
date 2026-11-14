"""spaCy PhraseMatcher based Skill Extractor with Week 2 Enhancements.

Fixes sentence-final punctuation tokenization defects (e.g., "LTE and 5G.",
"4G/5G.", "R.", "C."), enforces strict uppercase matching for acronyms of any
length (preventing "Siem Reap" -> "SIEM"), protects symbol boundaries (C vs C++
and C#), adds section & context awareness, and supports a high-throughput
tokenizer-only mode (`use_fast_tokenizer=True`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.preprocess_data import TextPreprocessor  # noqa: E402
from .dictionary_extractor import (  # noqa: E402
    COUNTRY_SPECIFIC_BROAD_SURFACE_MAP,
    DictionarySkillExtractor,
    NEGATION_CUE_PRECHECK_REGEX,
    SKILL_MODALITIES,
    SkillNormalizer,
    _EXCLUSION_ACTION_VERBS,
    _NEG_REQUIREMENT_VERBS,
    _PROHIBITION_CUES,
    _UNNECESSARY_CUES,
    classify_span_modality_regex,
    extract_non_taxonomy_prohibited_clauses_regex,
    infer_non_negated_modality,
    is_valid_skill_context,
    load_taxonomy_dataframe,
    normalize_phrase_casing_rule,
    parse_aliases_field,
)

_SPACY_NEG_TOKENS = frozenset({"not", "n't", "never", "neither", "nor", "without", "no"})
_CONTRASTIVE_LINKERS = frozenset({"but", "however", "whereas", "while", "yet", "though", "although"})


def trace_clause_governor(span_root: spacy.tokens.Token) -> list[spacy.tokens.Token]:
    """Walks head links from `span_root` while respecting clausal coordination splits.

    Prevents negation in an earlier conjunct (e.g. 'Python is not required,
    but SQL is essential') from bleeding across ', but' into a coordinated clause.
    """
    curr = span_root
    chain = [curr]
    while curr.head != curr:
        if curr.dep_ in {"conj", "advcl", "ccomp", "parataxis"} and curr.pos_ in {"VERB", "AUX", "ADJ"}:
            has_subj = any(ch.dep_ in {"nsubj", "nsubjpass"} for ch in curr.children)
            left_contrast = any(ch.lower_ in _CONTRASTIVE_LINKERS for ch in curr.lefts)
            head_contrast = any(ch.lower_ in _CONTRASTIVE_LINKERS for ch in curr.head.children)
            if has_subj or left_contrast or (curr.dep_ == "conj" and head_contrast):
                break
        curr = curr.head
        chain.append(curr)
    return chain


def classify_span_modality_spacy(
    doc: spacy.tokens.Doc,
    span: spacy.tokens.Span,
    source_section: str,
) -> dict[str, Any]:
    """Classifies modality and negation using spaCy's dependency tree."""
    if not doc.has_annotation("DEP"):
        fallback = classify_span_modality_regex(doc.text, span.start_char, span.end_char, source_section)
        fallback["polarity_evidence"]["analysis_method"] = "conservative_regex_fallback_no_dep_parse"
        return fallback

    sent_text = span.sent.text if span.sent is not None else doc.text
    chain = trace_clause_governor(span.root)
    dep_path = [f"{tok.text}({tok.dep_}:{tok.pos_})" for tok in chain]

    neg_cues: list[str] = []
    prohib_hits: list[str] = []
    unnec_hits: list[str] = []
    req_hits: list[str] = []
    act_hits: list[str] = []

    for node in chain:
        lemma_l = node.lemma_.lower()
        text_l = node.text.lower()
        if lemma_l in _PROHIBITION_CUES or text_l in _PROHIBITION_CUES:
            prohib_hits.append(node.text)
        if lemma_l in _UNNECESSARY_CUES or text_l in _UNNECESSARY_CUES:
            unnec_hits.append(node.text)
        if lemma_l in _NEG_REQUIREMENT_VERBS or text_l in _NEG_REQUIREMENT_VERBS:
            req_hits.append(node.text)
        if lemma_l in _EXCLUSION_ACTION_VERBS or text_l in _EXCLUSION_ACTION_VERBS:
            act_hits.append(node.text)

        for ch in node.children:
            ch_low = ch.lower_
            # Guard against benign boilerplate "no prior experience"
            if ch_low == "no" and node.lower_ == "experience" and not any(
                gc.lower_ in {"with", "in", "of"} for gc in node.children
            ):
                continue

            if ch.dep_ == "neg" or ch_low in _SPACY_NEG_TOKENS:
                # Ignore pre-sentential contrast linkers unrelated to node polarity
                if ch_low == "without" and node.pos_ == "NOUN" and not any(
                    sc == ch or sc in ch.subtree for sc in span
                ):
                    continue
                neg_cues.append(ch.text)

            if ch.dep_ in {"acomp", "attr", "oprd", "amod", "xcomp"}:
                if ch.lemma_.lower() in _UNNECESSARY_CUES and not any(gc.dep_ == "neg" for gc in ch.children):
                    unnec_hits.append(ch.text)
                if ch.lemma_.lower() in _NEG_REQUIREMENT_VERBS and ch.lemma_.lower() not in {"essential", "mandatory", "required", "necessary"}:
                    pass
                if ch.lemma_.lower() in _PROHIBITION_CUES:
                    prohib_hits.append(ch.text)
                if ch.lemma_.lower() in _NEG_REQUIREMENT_VERBS and not req_hits:
                    req_hits.append(ch.text)

    # Check direct siblings attached to preposition "without X"
    for tok in span:
        anc = tok.head
        while anc != anc.head:
            if anc.dep_ == "prep" and anc.lower_ == "without":
                neg_cues.append("without")
                break
            if anc in chain and anc.pos_ in {"VERB", "AUX"}:
                break
            anc = anc.head

    is_negated = False
    modality = infer_non_negated_modality(sent_text, source_section)
    sub_type: str | None = None

    if prohib_hits:
        is_negated = True
        modality = "excluded_or_prohibited"
        sub_type = "prohibited"
    elif unnec_hits and not neg_cues:
        is_negated = True
        modality = "negated_requirement"
        sub_type = "negated_requirement"
    elif neg_cues:
        is_negated = True
        if act_hits and not req_hits:
            modality = "excluded_or_prohibited"
            sub_type = "excluded"
        else:
            modality = "negated_requirement"
            sub_type = "negated_requirement"

    # Infer preference/requirement directly from affirmative governor when not negated
    if not is_negated and source_section not in {"required", "preferred"}:
        gov_lemmas = {n.lemma_.lower() for n in chain}
        gov_lemmas.update(ch.lemma_.lower() for n in chain for ch in n.children if ch.dep_ in {"acomp", "attr", "amod"})
        if gov_lemmas & {"prefer", "desirable", "optional", "bonus", "advantage"}:
            modality = "preferred"
        elif gov_lemmas & {"require", "must", "need", "essential", "mandatory", "necessary", "have"} and source_section != "intro_or_title":
            if any(w in gov_lemmas for w in ("require", "must", "need", "essential", "mandatory", "necessary")):
                modality = "required"

    gov_tok = chain[-1] if chain else span.root
    return {
        "modality": modality,
        "is_negated": is_negated,
        "polarity_evidence": {
            "analysis_method": "spacy_dependency_parser",
            "prohibition_type": sub_type,
            "governing_token": gov_tok.text,
            "governing_lemma": gov_tok.lemma_.lower(),
            "dependency_path": dep_path,
            "negation_cues": sorted(set(neg_cues + prohib_hits + unnec_hits)),
            "clause_text": sent_text.strip(),
        },
    }


def extract_non_taxonomy_prohibited_clauses_spacy(
    doc: spacy.tokens.Doc,
    extracted_spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detects general prohibitive clauses (e.g. 'avoid using deprecated technologies')."""
    if not doc.has_annotation("DEP"):
        return extract_non_taxonomy_prohibited_clauses_regex(doc.text, extracted_spans)

    findings: list[dict[str, Any]] = []
    for tok in doc:
        if tok.lemma_.lower() in _PROHIBITION_CUES and tok.pos_ in {"VERB", "ADJ"}:
            # Find governed object subtree
            candidate_objs: list[spacy.tokens.Token] = []
            for ch in tok.children:
                if ch.dep_ in {"dobj", "obj", "attr", "oprd"}:
                    candidate_objs.append(ch)
                elif ch.dep_ in {"xcomp", "ccomp", "prep"}:
                    for gc in ch.children:
                        if gc.dep_ in {"dobj", "obj", "pobj"}:
                            candidate_objs.append(gc)
            for obj in candidate_objs:
                subtree_tokens = sorted(obj.subtree, key=lambda t: t.i)
                s_char = subtree_tokens[0].idx
                e_char = subtree_tokens[-1].idx + len(subtree_tokens[-1].text)
                overlaps = any(
                    max(s_char, sp["character_start"]) < min(e_char, sp["character_end"])
                    for sp in extracted_spans
                )
                if overlaps:
                    continue
                phrase = doc.text[s_char:e_char].strip()
                findings.append({
                    "target_phrase": phrase,
                    "character_start": s_char,
                    "character_end": e_char,
                    "governing_verb": tok.lemma_.lower(),
                    "modality": "excluded_or_prohibited",
                    "is_negated": True,
                    "skill_id": None,
                    "canonical_skill": None,
                    "analysis_method": "spacy_dependency_parser",
                    "note": "Prohibited or deprecated non-taxonomy phrase captured without inventing a canonical skill ID.",
                })
    return findings


class SpacySkillExtractor:
    """Token-aware phrase matching extractor using spaCy with dependency negation."""

    def __init__(
        self,
        taxonomy_df_or_path: pd.DataFrame | str | Path | dict[str, Any],
        model_name: str = "en_core_web_sm",
        include_broad_country_mappings: bool = False,
        section_filter: Iterable[str] | None = None,
        enable_context_filters: bool = True,
        use_fast_tokenizer: bool = False,
        exclude_negated: bool = True,
    ) -> None:
        self.model_name = model_name
        self.nlp = spacy.load(model_name)
        self._configure_tokenizer()

        self.taxonomy_df = load_taxonomy_dataframe(taxonomy_df_or_path)
        self.include_broad_country_mappings = include_broad_country_mappings
        self.section_filter = set(section_filter) if section_filter is not None else None
        self.enable_context_filters = enable_context_filters
        self.use_fast_tokenizer = use_fast_tokenizer
        self.exclude_negated = exclude_negated
        self.has_dependency_parser = "parser" in self.nlp.pipe_names

        self.preprocessor = TextPreprocessor()
        self.normalizer = SkillNormalizer(
            self.taxonomy_df,
            include_broad_country_mappings=include_broad_country_mappings,
        )

        self.matcher_lower = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.matcher_exact = PhraseMatcher(self.nlp.vocab, attr="ORTH")
        self.pattern_metadata: dict[str, dict[str, Any]] = {}

        self._build_matchers()

    def _configure_tokenizer(self) -> None:
        """Configures custom tokenizer suffixes so trailing periods split cleanly.

        Resolves the Week 1 en_core_web_sm defect where sentence-final
        periods remained attached to alphanumeric tokens (e.g., "5G.", "4G/5G.",
        "R.", "C.") while keeping ".NET" and "Node.js" intact.
        """
        suffixes = list(self.nlp.Defaults.suffixes)
        trailing_period_rule = r"(?<=[A-Za-z0-9])\."
        if trailing_period_rule not in suffixes:
            suffixes.append(trailing_period_rule)
        self.nlp.tokenizer.suffix_search = spacy.util.compile_suffix_regex(suffixes).search

        rules = dict(self.nlp.tokenizer.rules)
        changed = False
        for abbr in ("R.", "C.", "r.", "c."):
            if abbr in rules:
                rules.pop(abbr, None)
                changed = True
        if changed:
            self.nlp.tokenizer.rules = rules

    def _build_matchers(self) -> None:
        """Populates PhraseMatchers with provenance-tagged canonical & alias docs."""
        lower_buckets: dict[str, list[spacy.tokens.Doc]] = {}
        exact_buckets: dict[str, list[spacy.tokens.Doc]] = {}
        seen_phrase_keys: set[tuple[str, str, bool]] = set()

        def register_phrase(
            phrase: str,
            case_sensitive: bool,
            skill_id: str,
            canonical: str,
            category: str,
            skill_type: str,
            is_alias: bool,
            match_method: str,
            mapping_status: str,
            mapping_confidence: float,
        ) -> None:
            dedup_key = (phrase if case_sensitive else phrase.lower(), skill_id, is_alias)
            if dedup_key in seen_phrase_keys:
                return
            seen_phrase_keys.add(dedup_key)

            bucket_key = f"{skill_id}::{match_method}::{int(is_alias)}"
            if bucket_key not in self.pattern_metadata:
                self.pattern_metadata[bucket_key] = {
                    "skill_id": skill_id,
                    "canonical_skill": canonical,
                    "skill_category": category,
                    "skill_type": skill_type,
                    "is_alias": is_alias,
                    "match_method": match_method,
                    "mapping_status": mapping_status,
                    "mapping_confidence": mapping_confidence,
                }

            doc_pattern = self.nlp.make_doc(phrase)
            if case_sensitive:
                exact_buckets.setdefault(bucket_key, []).append(doc_pattern)
            else:
                lower_buckets.setdefault(bucket_key, []).append(doc_pattern)

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
                is_short = match_method == "context_verified_short_token"
                confidence = 0.90 if is_short else (0.95 if is_alias else 1.0)
                register_phrase(
                    phrase=phrase,
                    case_sensitive=case_sensitive,
                    skill_id=skill_id,
                    canonical=canonical,
                    category=category,
                    skill_type=skill_type,
                    is_alias=is_alias,
                    match_method=match_method,
                    mapping_status="MAPPED",
                    mapping_confidence=confidence,
                )

        if self.include_broad_country_mappings:
            for item in COUNTRY_SPECIFIC_BROAD_SURFACE_MAP:
                target = self.normalizer.skill_by_id.get(item["skill_id"])
                if target is not None:
                    register_phrase(
                        phrase=item["phrase"],
                        case_sensitive=item["case_sensitive"],
                        skill_id=target["skill_id"],
                        canonical=target["canonical_skill"],
                        category=target["skill_category"],
                        skill_type=target["skill_type"],
                        is_alias=True,
                        match_method="country_specific_broad_match",
                        mapping_status="BROAD_COUNTRY_MAPPING",
                        mapping_confidence=0.80,
                    )

        for bucket_key, docs in lower_buckets.items():
            self.matcher_lower.add(bucket_key, docs)

        for bucket_key, docs in exact_buckets.items():
            self.matcher_exact.add(bucket_key, docs)

    def _build_doc(self, norm_text: str) -> spacy.tokens.Doc:
        """Uses fast tokenizer on clean texts, or full parser when negation cues appear."""
        needs_dep_parse = (not self.use_fast_tokenizer) or bool(NEGATION_CUE_PRECHECK_REGEX.search(norm_text))
        if needs_dep_parse and self.has_dependency_parser:
            return self.nlp(norm_text)
        return self.nlp.make_doc(norm_text)

    def _extract_with_doc(
        self,
        text: str,
        norm_text: str,
        doc: spacy.tokens.Doc,
        include_negated: bool = False,
    ) -> list[dict[str, Any]]:
        sections = self.preprocessor.segment_job_sections(text)
        raw_matches: list[dict[str, Any]] = []
        span_lookup: dict[tuple[int, int, str], spacy.tokens.Span] = {}

        for matcher in (self.matcher_lower, self.matcher_exact):
            for match_id, start_idx, end_idx in matcher(doc):
                span = doc[start_idx:end_idx]
                bucket_key = self.nlp.vocab.strings[match_id]
                meta = self.pattern_metadata.get(bucket_key, {})

                c_start = span.start_char
                c_end = span.end_char
                matched_text = text[c_start:c_end]

                assert len(matched_text) == (c_end - c_start), (
                    f"Offset alignment error: expected {c_end - c_start}, got {len(matched_text)}"
                )

                canonical_skill = meta.get("canonical_skill", "")
                if self.enable_context_filters and not is_valid_skill_context(
                    norm_text, c_start, c_end, matched_text, canonical_skill
                ):
                    continue

                source_section = self.preprocessor.get_section_for_span(text, c_start, c_end, sections)
                if self.section_filter is not None and source_section not in self.section_filter:
                    continue

                span_lookup[(c_start, c_end, canonical_skill)] = span
                raw_matches.append({
                    "skill_mention": matched_text,
                    "matched_text": matched_text,
                    "canonical_skill": canonical_skill,
                    "skill_id": meta.get("skill_id", ""),
                    "skill_category": meta.get("skill_category", ""),
                    "skill_type": meta.get("skill_type", ""),
                    "character_start": c_start,
                    "character_end": c_end,
                    "match_length": c_end - c_start,
                    "is_alias": meta.get("is_alias", False),
                    "extractor_name": "SpacySkillExtractor",
                    "extraction_method": "spacy_phrasematcher",
                    "match_method": meta.get("match_method", "exact_match"),
                    "mapping_status": meta.get("mapping_status", "MAPPED"),
                    "mapping_confidence": meta.get("mapping_confidence", 1.0),
                    "source_section": source_section,
                })

        resolved = DictionarySkillExtractor._resolve_overlaps(raw_matches)
        output: list[dict[str, Any]] = []
        has_dep = doc.has_annotation("DEP")
        for rec in resolved:
            if not has_dep and rec["source_section"] in {"required", "preferred"}:
                polarity = {
                    "modality": rec["source_section"],
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
                sp_obj = span_lookup[(rec["character_start"], rec["character_end"], rec["canonical_skill"])]
                polarity = classify_span_modality_spacy(doc, sp_obj, rec["source_section"])

            rec["modality"] = polarity["modality"]
            rec["is_negated"] = polarity["is_negated"]
            rec["polarity_evidence"] = polarity["polarity_evidence"]

            if self.exclude_negated and not include_negated and rec["is_negated"]:
                continue
            output.append(rec)

        return output

    def extract(self, text: Any, include_negated: bool = False) -> list[dict[str, Any]]:
        """Extracts normalized skill entities preserving offsets, modality, and negation."""
        if not isinstance(text, str) or not text.strip():
            return []

        norm_text = self.preprocessor.normalize_preserving_offsets(text)["normalized_text"]
        doc = self._build_doc(norm_text)
        return self._extract_with_doc(text, norm_text, doc, include_negated=include_negated)

    def analyze_sentences_and_negation(self, text: Any) -> dict[str, Any]:
        """Runs dependency-aware sentence modality & non-taxonomy prohibition analysis."""
        if not isinstance(text, str) or not text.strip():
            return {
                "positive_spans": [],
                "negated_spans": [],
                "all_spans": [],
                "non_taxonomy_prohibited_clauses": [],
                "analysis_method": "spacy_dependency_parser" if self.has_dependency_parser else "conservative_regex_fallback_parser_unavailable",
            }
        norm_text = self.preprocessor.normalize_preserving_offsets(text)["normalized_text"]
        doc = self.nlp(norm_text) if self.has_dependency_parser else self.nlp.make_doc(norm_text)
        all_spans = self._extract_with_doc(text, norm_text, doc, include_negated=True)
        positive_spans = [s for s in all_spans if not s["is_negated"]]
        negated_spans = [s for s in all_spans if s["is_negated"]]
        non_tax = extract_non_taxonomy_prohibited_clauses_spacy(doc, all_spans)
        return {
            "positive_spans": positive_spans,
            "negated_spans": negated_spans,
            "all_spans": all_spans,
            "non_taxonomy_prohibited_clauses": non_tax,
            "analysis_method": "spacy_dependency_parser" if doc.has_annotation("DEP") else "conservative_regex_fallback_parser_unavailable",
        }

    @staticmethod
    def _resolve_overlaps(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return DictionarySkillExtractor._resolve_overlaps(matches)
