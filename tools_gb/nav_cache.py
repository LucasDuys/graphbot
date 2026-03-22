"""Navigation sequence caching via the knowledge graph.

Stores successful browser navigation sequences as Skill nodes in the graph
so they can be retrieved and replayed for similar future tasks. Each cached
sequence records the target URL, the ordered list of browser actions, and
an extraction template describing what data to pull from the final page.

Lookup is performed by exact URL or by domain prefix, enabling reuse of
navigation patterns across pages on the same site.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from graph.store import GraphStore

logger = logging.getLogger(__name__)

# Prefix used for Skill node names that represent navigation sequences.
_NAV_SKILL_PREFIX: str = "nav:"


@dataclass
class NavigationSequence:
    """A cached browser navigation sequence.

    Attributes:
        url: The target URL this sequence navigates to.
        action_sequence: Ordered list of browser actions. Each action is a
            dict with at minimum an ``action`` key (navigate, click, fill,
            extract_text, screenshot) plus action-specific parameters.
        extracted_data_template: CSS selector mapping describing what data
            to extract from the final page state. Keys are logical field
            names, values are CSS selectors.
    """

    url: str
    action_sequence: list[dict[str, Any]]
    extracted_data_template: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        """Serialize to a dict suitable for graph storage.

        Lists and dicts are JSON-encoded to fit the STRING column type in
        the Kuzu schema.
        """
        return {
            "url": self.url,
            "action_sequence": json.dumps(self.action_sequence),
            "extracted_data_template": json.dumps(self.extracted_data_template),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NavigationSequence:
        """Deserialize from a graph node property dict.

        Handles both raw dicts (already parsed) and JSON strings (as stored
        in the graph).
        """
        action_seq_raw = data.get("action_sequence", "[]")
        if isinstance(action_seq_raw, str):
            action_seq = json.loads(action_seq_raw)
        else:
            action_seq = action_seq_raw

        template_raw = data.get("extracted_data_template", "{}")
        if isinstance(template_raw, str):
            template = json.loads(template_raw)
        else:
            template = template_raw

        return cls(
            url=str(data.get("url", "")),
            action_sequence=action_seq,
            extracted_data_template=template,
        )


class NavigationCache:
    """Reads and writes navigation sequences as Skill nodes in the graph.

    Each navigation sequence is stored as a Skill node with:
    - name: ``nav:<url>`` (used as a lookup key)
    - domain: ``browser``
    - url: target URL
    - action_sequence: JSON-encoded list of actions
    - extracted_data_template: JSON-encoded selector map
    - description: human-readable summary

    Args:
        store: An initialized GraphStore instance.
    """

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    def store(self, sequence: NavigationSequence) -> str:
        """Store a navigation sequence in the graph.

        If a sequence for the same URL already exists, it is updated
        in-place (the same Skill node id is returned).

        Args:
            sequence: The navigation sequence to cache.

        Returns:
            The Skill node id.
        """
        existing_id = self._find_skill_id_by_url(sequence.url)
        serialized = sequence.to_dict()

        if existing_id is not None:
            self._store.update_node("Skill", existing_id, {
                "action_sequence": serialized["action_sequence"],
                "extracted_data_template": serialized["extracted_data_template"],
                "description": self._build_description(sequence),
            })
            logger.info("Updated cached navigation for %s (id=%s)", sequence.url, existing_id)
            return existing_id

        skill_id = self._store.create_node("Skill", {
            "name": f"{_NAV_SKILL_PREFIX}{sequence.url}",
            "description": self._build_description(sequence),
            "domain": "browser",
            "url": sequence.url,
            "action_sequence": serialized["action_sequence"],
            "extracted_data_template": serialized["extracted_data_template"],
            "path": "",
        })
        logger.info("Cached new navigation sequence for %s (id=%s)", sequence.url, skill_id)
        return skill_id

    def find_by_url(self, url: str) -> NavigationSequence | None:
        """Find a cached navigation sequence by exact URL.

        Args:
            url: The target URL to search for.

        Returns:
            The cached NavigationSequence, or None if not found.
        """
        name = f"{_NAV_SKILL_PREFIX}{url}"
        rows = self._store.query(
            "MATCH (s:Skill) WHERE s.name = $name RETURN s.*",
            params={"name": name},
        )
        if not rows:
            return None

        props = self._unprefix_row(rows[0])
        return NavigationSequence.from_dict(props)

    def find_by_domain(self, domain: str) -> list[NavigationSequence]:
        """Find all cached navigation sequences for a given domain.

        Args:
            domain: The hostname to search for (e.g. ``shop.example.com``).

        Returns:
            List of matching NavigationSequence objects (may be empty).
        """
        prefix = f"{_NAV_SKILL_PREFIX}https://{domain}"
        rows = self._store.query(
            "MATCH (s:Skill) WHERE s.name STARTS WITH $prefix RETURN s.*",
            params={"prefix": prefix},
        )

        results: list[NavigationSequence] = []
        for row in rows:
            props = self._unprefix_row(row)
            try:
                results.append(NavigationSequence.from_dict(props))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping malformed navigation skill: %s", exc)

        return results

    def delete(self, url: str) -> bool:
        """Delete a cached navigation sequence by URL.

        Args:
            url: The target URL whose cached sequence to remove.

        Returns:
            True if a sequence was found and deleted, False otherwise.
        """
        skill_id = self._find_skill_id_by_url(url)
        if skill_id is None:
            return False

        return self._store.delete_node("Skill", skill_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_skill_id_by_url(self, url: str) -> str | None:
        """Locate the Skill node id for a navigation sequence by URL."""
        name = f"{_NAV_SKILL_PREFIX}{url}"
        rows = self._store.query(
            "MATCH (s:Skill) WHERE s.name = $name RETURN s.id",
            params={"name": name},
        )
        if not rows:
            return None
        # Column name is "s.id"
        first_row = rows[0]
        return str(first_row.get("s.id", ""))

    @staticmethod
    def _unprefix_row(row: dict[str, Any]) -> dict[str, Any]:
        """Strip the ``s.`` prefix from Kuzu column names."""
        return {
            (k.split(".", 1)[1] if "." in k else k): v
            for k, v in row.items()
        }

    @staticmethod
    def _build_description(sequence: NavigationSequence) -> str:
        """Build a human-readable description for the Skill node."""
        action_count = len(sequence.action_sequence)
        parsed = urlparse(sequence.url)
        domain = parsed.netloc or parsed.path
        return f"Browser navigation to {domain} ({action_count} steps)"
