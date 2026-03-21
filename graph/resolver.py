"""Entity resolution -- matching text mentions to existing graph entities."""

from __future__ import annotations

import math
from collections import Counter

import Levenshtein

from graph.schema import NODE_TYPES
from graph.store import GraphStore

# Tables that use "name" as the display field, and File uses "path".
_NAME_TABLES: list[str] = [
    nt.name for nt in NODE_TYPES if "name" in nt.properties and nt.name != "File"
]
_FILE_TABLE: str = "File"

# Tables that have a "description" field (for keyword matching).
_DESCRIPTION_TABLES: set[str] = {
    nt.name for nt in NODE_TYPES if "description" in nt.properties
}

# Tables that have a "relationship" field (Contact).
_RELATIONSHIP_TABLES: set[str] = {
    nt.name for nt in NODE_TYPES if "relationship" in nt.properties
}

# Tables that have a "type" field (for keyword matching).
_TYPE_TABLES: set[str] = {
    nt.name for nt in NODE_TYPES if "type" in nt.properties and nt.name != "File"
}


class EntityResolver:
    """Resolves text mentions to existing graph entities using 3-layer matching."""

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    def resolve(
        self, mention: str, top_k: int = 5, edit_threshold: float = 0.8
    ) -> list[tuple[str, float]]:
        """Resolve a text mention to entity IDs with confidence scores.

        Returns list of (entity_id, confidence) sorted by confidence descending.

        Layers:
        1. Exact normalized match (lowercase, strip) -- confidence 1.0
        2. Levenshtein ratio match (threshold configurable) -- confidence = ratio
        3. BM25-style keyword match on name + description -- confidence = score (0-1 normalized)

        No LLM calls -- purely algorithmic.
        """
        normalized = mention.lower().strip()
        if not normalized:
            return []

        # Collect all candidate entities: list of (id, display_text, extra_text)
        candidates = self._load_candidates()

        # Best confidence per entity id (across all layers).
        best: dict[str, float] = {}
        exact_ids: set[str] = set()

        # Layer 1: exact normalized match
        for eid, display, _extra in candidates:
            if display.lower().strip() == normalized:
                best[eid] = 1.0
                exact_ids.add(eid)

        # Layer 2: Levenshtein ratio (skip entities already matched exactly)
        for eid, display, _extra in candidates:
            if eid in exact_ids:
                continue
            ratio = Levenshtein.ratio(normalized, display.lower().strip())
            if ratio >= edit_threshold:
                if ratio > best.get(eid, 0.0):
                    best[eid] = ratio

        # Layer 3: BM25-style keyword match
        mention_words = normalized.split()
        if mention_words:
            # Build document frequency counts across all candidate text
            doc_count = len(candidates)
            df: Counter[str] = Counter()
            for _eid, display, extra in candidates:
                words_in_doc = set((display + " " + extra).lower().split())
                for w in words_in_doc:
                    df[w] += 1

            for eid, display, extra in candidates:
                doc_text = (display + " " + extra).lower()
                doc_words = set(doc_text.split())
                matching = sum(1 for w in mention_words if w in doc_words)
                if matching == 0:
                    continue

                # BM25-inspired: weight by IDF
                score = 0.0
                for w in mention_words:
                    if w in doc_words:
                        idf = math.log((doc_count - df[w] + 0.5) / (df[w] + 0.5) + 1.0)
                        score += idf

                # Normalize: max possible score is if all words matched with max IDF
                max_score = 0.0
                for w in mention_words:
                    idf = math.log((doc_count - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5) + 1.0)
                    max_score += idf

                if max_score > 0:
                    normalized_score = score / max_score
                else:
                    normalized_score = matching / len(mention_words)

                if normalized_score > 0.3:
                    if normalized_score > best.get(eid, 0.0):
                        best[eid] = normalized_score

        # Sort by confidence descending, then return top_k
        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _load_candidates(self) -> list[tuple[str, str, str]]:
        """Load all candidate entities from the graph.

        Returns list of (entity_id, display_text, extra_text) where:
        - display_text is the name (or path for File)
        - extra_text is description + relationship + type fields concatenated
        """
        candidates: list[tuple[str, str, str]] = []

        # Named tables (User, Project, Service, Contact, Skill, etc.)
        for table in _NAME_TABLES:
            rows = self._store.query(f"MATCH (n:{table}) RETURN n.id, n.name")
            extra_fields = []
            if table in _DESCRIPTION_TABLES:
                extra_fields.append("description")
            if table in _RELATIONSHIP_TABLES:
                extra_fields.append("relationship")
            if table in _TYPE_TABLES:
                extra_fields.append("type")

            if extra_fields:
                extra_cols = ", ".join(f"n.{f}" for f in extra_fields)
                rows = self._store.query(
                    f"MATCH (n:{table}) RETURN n.id, n.name, {extra_cols}"
                )
                for row in rows:
                    eid = str(row["n.id"])
                    name = str(row.get("n.name") or "")
                    extra_parts = []
                    for f in extra_fields:
                        val = row.get(f"n.{f}")
                        if val:
                            extra_parts.append(str(val))
                    candidates.append((eid, name, " ".join(extra_parts)))
            else:
                for row in rows:
                    eid = str(row["n.id"])
                    name = str(row.get("n.name") or "")
                    candidates.append((eid, name, ""))

        # File table uses "path" as display text
        file_rows = self._store.query(
            f"MATCH (n:{_FILE_TABLE}) RETURN n.id, n.path, n.description"
        )
        for row in file_rows:
            eid = str(row["n.id"])
            path = str(row.get("n.path") or "")
            desc = str(row.get("n.description") or "")
            candidates.append((eid, path, desc))

        return candidates
