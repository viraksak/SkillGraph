"""Test and verify the skill extraction pipeline logic before building the notebook.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

ROOT_DIR = Path(__file__).resolve().parent.parent

# Load taxonomy
with open(ROOT_DIR / "outputs" / "exports" / "skill_taxonomy.json", "r", encoding="utf-8") as f:
    TAXONOMY = json.load(f)

# Load raw jobs
with open(ROOT_DIR / "data" / "raw" / "synthetic_job_postings.json", "r", encoding="utf-8") as f:
    RAW_DATA = json.load(f)["data"]

# Load ground truth
with open(ROOT_DIR / "data" / "annotations" / "synthetic_job_postings_ground_truth.json", "r", encoding="utf-8") as f:
    GROUND_TRUTH = {item["job_id"]: set(item["ground_truth_skills"]) for item in json.load(f)["annotations"]}

# 1. Regex Matcher Implementation
class RegexSkillExtractor:
    def __init__(self, taxonomy: dict):
        self.taxonomy = taxonomy
        self.alias_to_canonical = {}
        self.patterns = []
        self._compile_patterns()

    def _compile_patterns(self):
        for canonical, info in self.taxonomy.items():
            for alias in info["aliases"]:
                self.alias_to_canonical[alias.lower()] = canonical
                
                # Tricky short / special skills handling
                if alias.lower() in ["c", "r"]:
                    # Exact uppercase letter, not part of words, acronyms like R&D, or hyphenated prefixes
                    pattern_str = rf"(?<![A-Za-z0-9&/\-_]){re.escape(alias.upper())}(?![A-Za-z0-9&/\-_])"
                    # Additional context filter for Vitamin C, C-suite
                    pattern = re.compile(pattern_str)
                    self.patterns.append((pattern, canonical, alias, True)) # case-sensitive
                elif alias.lower() == "go":
                    # Case-sensitive "Go" or "Golang" (case-insensitive for golang)
                    pattern_str = r"(?<![A-Za-z0-9])Go(?![A-Za-z0-9])"
                    pattern = re.compile(pattern_str)
                    self.patterns.append((pattern, canonical, alias, True))
                elif alias.lower() == "golang":
                    pattern_str = r"\bgolang\b"
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    self.patterns.append((pattern, canonical, alias, False))
                elif "+" in alias or "/" in alias or "." in alias:
                    # Symbols require custom non-alphanumeric boundary
                    pattern_str = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    self.patterns.append((pattern, canonical, alias, False))
                else:
                    # Standard word boundary
                    pattern_str = rf"\b{re.escape(alias)}\b"
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    self.patterns.append((pattern, canonical, alias, False))

    def extract(self, text: str) -> list[dict]:
        extractions = []
        # Pre-clean false positive contexts for C and Go if present
        # e.g., 'Vitamin C', 'C-suite', 'ready to go', 'go the extra mile'
        for pattern, canonical, alias, is_case_sensitive in self.patterns:
            matches = list(pattern.finditer(text))
            for match in matches:
                start, end = match.span()
                matched_text = match.group()
                
                # Context check for single letter C: avoid "Vitamin C", "C-suite", "C-level"
                if canonical == "C" and matched_text == "C":
                    context_window = text[max(0, start - 10):min(len(text), end + 10)]
                    if "vitamin" in context_window.lower() or "suite" in context_window.lower() or "level" in context_window.lower():
                        continue
                
                # Context check for Go: avoid "go to", "ready to go", "go the"
                if canonical == "Go" and matched_text == "Go":
                    context_window = text[max(0, start - 15):min(len(text), end + 15)].lower()
                    if "to go" in context_window or "ready to" in context_window or "go the" in context_window or "go from" in context_window:
                        continue

                extractions.append({
                    "canonical_skill": canonical,
                    "matched_text": matched_text,
                    "start": start,
                    "end": end,
                    "alias": alias
                })
        
        # Deduplicate by canonical_skill per document
        return extractions

# 2. spaCy PhraseMatcher Implementation
class SpacySkillExtractor:
    def __init__(self, taxonomy: dict):
        self.nlp = spacy.load("en_core_web_sm")
        self.taxonomy = taxonomy
        self.alias_to_canonical = {}
        
        # We maintain two matchers:
        # Case-insensitive for regular multi-word and distinctive terms
        self.matcher_lower = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        # Case-sensitive for short ambiguous tokens like C, R, Go
        self.matcher_exact = PhraseMatcher(self.nlp.vocab, attr="ORTH")
        
        self._build_matchers()

    def _build_matchers(self):
        for canonical, info in self.taxonomy.items():
            for alias in info["aliases"]:
                self.alias_to_canonical[alias] = canonical
                self.alias_to_canonical[alias.lower()] = canonical
                
                if alias.lower() in ["c", "r", "go"]:
                    doc = self.nlp.make_doc(alias.upper() if alias.lower() != "go" else "Go")
                    self.matcher_exact.add(canonical, [doc])
                else:
                    doc = self.nlp.make_doc(alias)
                    self.matcher_lower.add(canonical, [doc])

    def extract(self, text: str) -> list[dict]:
        doc = self.nlp(text)
        extractions = []
        
        # Match lower
        matches_lower = self.matcher_lower(doc)
        for match_id, start, end in matches_lower:
            span = doc[start:end]
            canonical = self.nlp.vocab.strings[match_id]
            extractions.append({
                "canonical_skill": canonical,
                "matched_text": span.text,
                "start": span.start_char,
                "end": span.end_char
            })

        # Match exact
        matches_exact = self.matcher_exact(doc)
        for match_id, start, end in matches_exact:
            span = doc[start:end]
            canonical = self.nlp.vocab.strings[match_id]
            matched_text = span.text
            
            # False positive context filters
            if canonical == "C" and matched_text == "C":
                window = text[max(0, span.start_char - 10):min(len(text), span.end_char + 10)].lower()
                if "vitamin" in window or "suite" in window or "level" in window:
                    continue
            if canonical == "Go" and matched_text == "Go":
                window = text[max(0, span.start_char - 15):min(len(text), span.end_char + 15)].lower()
                if "to go" in window or "ready to" in window or "go the" in window or "go from" in window:
                    continue

            extractions.append({
                "canonical_skill": canonical,
                "matched_text": span.text,
                "start": span.start_char,
                "end": span.end_char
            })

        return extractions


def evaluate(predictions: dict[str, set[str]], ground_truth: dict[str, set[str]]) -> dict:
    per_doc_metrics = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for job_id, true_skills in ground_truth.items():
        pred_skills = predictions.get(job_id, set())
        tp = len(pred_skills & true_skills)
        fp = len(pred_skills - true_skills)
        fn = len(true_skills - pred_skills)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        per_doc_metrics.append({
            "job_id": job_id,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        })
        
        total_tp += tp
        total_fp += total_fp if False else fp
        total_fn += fn

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    df_doc = pd.DataFrame(per_doc_metrics)
    macro_prec = df_doc["precision"].mean()
    macro_rec = df_doc["recall"].mean()
    macro_f1 = df_doc["f1"].mean()

    return {
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f1": micro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn
    }

def main():
    print("Testing Regex Extractor...")
    regex_extractor = RegexSkillExtractor(TAXONOMY)
    regex_preds = {}
    for job in RAW_DATA:
        extracted = regex_extractor.extract(job["description"])
        regex_preds[job["job_id"]] = {item["canonical_skill"] for item in extracted}

    regex_metrics = evaluate(regex_preds, GROUND_TRUTH)
    print("Regex Metrics:", regex_metrics)

    print("\nTesting spaCy PhraseMatcher...")
    spacy_extractor = SpacySkillExtractor(TAXONOMY)
    spacy_preds = {}
    for job in RAW_DATA:
        extracted = spacy_extractor.extract(job["description"])
        spacy_preds[job["job_id"]] = {item["canonical_skill"] for item in extracted}

    spacy_metrics = evaluate(spacy_preds, GROUND_TRUTH)
    print("spaCy Metrics:", spacy_metrics)

if __name__ == "__main__":
    main()

