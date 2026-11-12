"""SkillGraph Skill Extraction Module.

Provides dictionary-based, spaCy PhraseMatcher, and evaluation pipelines
for technical and transversal skill extraction from job descriptions.
"""

from .dictionary_extractor import DictionarySkillExtractor
from .spacy_extractor import SpacySkillExtractor
from .evaluator import SkillExtractionEvaluator

__all__ = [
    "DictionarySkillExtractor",
    "SpacySkillExtractor",
    "SkillExtractionEvaluator",
]

