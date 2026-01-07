"""Query Processor - Preprocessing and expansion for RAG queries.

Features:
- Abbreviation expansion (DW -> Death Wave)
- Synonym expansion for better recall
- Typo correction
- Filter extraction (flair:Guide)
- Query normalization
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ProcessedQuery:
    """Result of query processing."""
    original: str
    normalized: str
    expanded: str
    keywords: List[str]
    filters: Dict[str, str] = field(default_factory=dict)
    expansions_applied: List[str] = field(default_factory=list)


class QueryProcessor:
    """
    Processes search queries for better RAG retrieval.

    Handles:
    - Abbreviation expansion (DW -> Death Wave)
    - Typo correction
    - Filter extraction (flair:Guide, type:wiki)
    - Query normalization
    """

    def __init__(self, glossary_path: str = None):
        """
        Initialize query processor.

        Args:
            glossary_path: Path to glossary YAML file
        """
        self.glossary_path = glossary_path or self._find_glossary()
        self.glossary = self._load_glossary()

        # Build lookup tables
        self.abbreviations = self.glossary.get("abbreviations", {})
        self.synonyms = self.glossary.get("synonyms", {})
        self.typos = self.glossary.get("typos", {})
        self.expansions = self.glossary.get("expansions", {})
        self.flair_aliases = self.glossary.get("flair_aliases", {})

        # Build reverse lookup for abbreviations (case-insensitive)
        self.abbrev_lower = {k.lower(): v for k, v in self.abbreviations.items()}

        # Build typo patterns for efficient matching
        self.typo_patterns = {k.lower(): v for k, v in self.typos.items()}

    def _find_glossary(self) -> str:
        """Find the glossary file."""
        possible_paths = [
            Path("config/tower_glossary.yaml"),
            Path("../config/tower_glossary.yaml"),
            Path(__file__).parent.parent.parent / "config" / "tower_glossary.yaml"
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        return ""

    def _load_glossary(self) -> dict:
        """Load glossary from YAML file."""
        if not self.glossary_path or not Path(self.glossary_path).exists():
            print("[QueryProcessor] No glossary found, using defaults")
            return self._default_glossary()

        try:
            with open(self.glossary_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[QueryProcessor] Error loading glossary: {e}")
            return self._default_glossary()

    def _default_glossary(self) -> dict:
        """Default glossary if file not found."""
        return {
            "abbreviations": {
                "DW": "Death Wave",
                "BH": "Black Hole",
                "GT": "Golden Tower",
                "UW": "Ultimate Weapon",
                "ILM": "Inner Land Mines",
                "CPH": "Coins Per Hour",
                "eHP": "effective Health Points"
            },
            "synonyms": {},
            "typos": {},
            "expansions": {}
        }

    def process(self, query: str) -> ProcessedQuery:
        """
        Process a query for RAG retrieval.

        Args:
            query: Raw user query

        Returns:
            ProcessedQuery with normalized and expanded query
        """
        original = query.strip()

        # Extract filters first
        normalized, filters = self._extract_filters(original)

        # Fix typos
        normalized = self._fix_typos(normalized)

        # Expand abbreviations
        expanded, expansions = self._expand_abbreviations(normalized)

        # Extract keywords
        keywords = self._extract_keywords(expanded)

        return ProcessedQuery(
            original=original,
            normalized=normalized,
            expanded=expanded,
            keywords=keywords,
            filters=filters,
            expansions_applied=expansions
        )

    def _extract_filters(self, query: str) -> Tuple[str, Dict[str, str]]:
        """
        Extract filter patterns from query.

        Patterns:
        - flair:Guide
        - type:wiki
        - score:>100

        Returns:
            Tuple of (query without filters, filters dict)
        """
        filters = {}

        # Pattern: key:value or key:"value with spaces"
        filter_pattern = r'(\w+):(?:"([^"]+)"|(\S+))'

        matches = re.findall(filter_pattern, query)
        for key, quoted_val, unquoted_val in matches:
            value = quoted_val or unquoted_val
            filters[key.lower()] = value

        # Remove filters from query
        cleaned = re.sub(filter_pattern, '', query).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace

        return cleaned, filters

    def _fix_typos(self, query: str) -> str:
        """Fix common typos in query."""
        query_lower = query.lower()

        for typo, correction in self.typo_patterns.items():
            if typo in query_lower:
                # Case-insensitive replacement
                pattern = re.compile(re.escape(typo), re.IGNORECASE)
                query = pattern.sub(correction, query)

        return query

    def _expand_abbreviations(self, query: str) -> Tuple[str, List[str]]:
        """
        Expand abbreviations in query.

        Example: "best DW build" -> "best Death Wave build"

        Returns:
            Tuple of (expanded query, list of expansions applied)
        """
        expansions_applied = []
        words = query.split()
        expanded_words = []

        for word in words:
            # Check if word is an abbreviation (case-insensitive)
            word_lower = word.lower()
            word_clean = re.sub(r'[^\w]', '', word_lower)  # Remove punctuation

            if word_clean in self.abbrev_lower:
                expansion = self.abbrev_lower[word_clean]
                expanded_words.append(expansion)
                expansions_applied.append(f"{word} -> {expansion}")
            else:
                expanded_words.append(word)

        return ' '.join(expanded_words), expansions_applied

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        stop_words = {
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'can', 'may', 'might', 'must', 'shall', 'this', 'that',
            'these', 'those', 'what', 'which', 'who', 'whom', 'how', 'when',
            'where', 'why', 'if', 'then', 'so', 'just', 'about', 'into', 'over',
            'best', 'good', 'need', 'want', 'looking', 'help', 'please'
        }

        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) >= 2]

        return keywords

    def expand_with_synonyms(self, query: str) -> str:
        """
        Expand query with synonyms for better recall.

        Example: "best damage build" -> "best damage dmg dps attack power build"
        """
        words = query.lower().split()
        expanded_parts = [query]  # Start with original

        for word in words:
            # Check if word has synonyms
            if word in self.synonyms:
                # Add synonyms
                syns = self.synonyms[word]
                expanded_parts.extend(syns[:2])  # Add top 2 synonyms

            # Also check if word IS a synonym
            for main_term, syn_list in self.synonyms.items():
                if word in [s.lower() for s in syn_list]:
                    expanded_parts.append(main_term)
                    break

        return ' '.join(expanded_parts)

    def get_search_variants(self, query: str) -> List[str]:
        """
        Generate multiple search variants for a query.

        Returns list of query variants to search.
        """
        processed = self.process(query)
        variants = [
            processed.original,      # Original query
            processed.expanded,      # With abbreviations expanded
        ]

        # Add synonym-expanded version if different
        syn_expanded = self.expand_with_synonyms(processed.expanded)
        if syn_expanded != processed.expanded:
            variants.append(syn_expanded)

        # Check for game-specific expansions
        for term, expansions in self.expansions.items():
            if term.lower() in processed.expanded.lower():
                # Add a variant with expansion terms
                variants.extend(expansions[:2])

        # Deduplicate while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            v_lower = v.lower()
            if v_lower not in seen:
                seen.add(v_lower)
                unique_variants.append(v)

        return unique_variants

    def suggest_corrections(self, query: str) -> Optional[str]:
        """
        Suggest corrections for a query.

        Returns corrected query or None if no corrections.
        """
        processed = self.process(query)

        if processed.expansions_applied or processed.normalized != processed.original:
            return processed.expanded

        return None


# Convenience function
def process_query(query: str) -> ProcessedQuery:
    """Process a query using default processor."""
    processor = QueryProcessor()
    return processor.process(query)


if __name__ == "__main__":
    # Test the processor
    processor = QueryProcessor()

    test_queries = [
        "best DW build",
        "how to farm CPH",
        "ILM vs BH for T15",
        "flair:Guide death wave",
        "deathwave strategy",
        "what is eHP",
        "optimal GT setup for endgame"
    ]

    print("Query Processing Tests")
    print("=" * 60)

    for query in test_queries:
        result = processor.process(query)
        print(f"\nOriginal: {result.original}")
        print(f"Expanded: {result.expanded}")
        print(f"Keywords: {result.keywords}")
        if result.filters:
            print(f"Filters: {result.filters}")
        if result.expansions_applied:
            print(f"Expansions: {result.expansions_applied}")

        variants = processor.get_search_variants(query)
        print(f"Search variants: {variants}")
