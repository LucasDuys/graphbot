"""Web tools for GraphBot DAG leaf execution."""

import asyncio
import html
import logging
import re
from typing import Any

import httpx
from readability import Document

logger = logging.getLogger(__name__)

USER_AGENT = "GraphBot/1.0"


class WebTool:
    """Web scraping and search tools for DAG leaf execution."""

    def __init__(self, max_tokens: int = 4000, timeout: int = 15) -> None:
        self._max_tokens = max_tokens
        self._timeout = timeout

    async def fetch(self, url: str) -> dict[str, Any]:
        """Fetch a URL and extract clean text content.

        Uses readability-lxml for content extraction.
        Returns: {success, content, title, url, tokens, truncated, error}
        """
        if not url.startswith(("http://", "https://")):
            return {"success": False, "error": "Invalid URL scheme", "url": url}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.get(
                    url,
                    headers={"User-Agent": USER_AGENT},
                    follow_redirects=True,
                )
                r.raise_for_status()

            doc = Document(r.text)
            title = doc.title() or "Untitled"
            html_content = doc.summary()
            markdown = self._html_to_markdown(html_content)

            truncated = len(markdown) > self._max_tokens * 4  # rough char estimate
            if truncated:
                markdown = markdown[: self._max_tokens * 4] + "\n\n[... truncated ...]"

            token_est = max(1, int(len(markdown.split()) * 1.3))

            return {
                "success": True,
                "content": markdown,
                "title": title,
                "url": str(r.url),
                "tokens": token_est,
                "truncated": truncated,
            }
        except httpx.HTTPStatusError as exc:
            return {"success": False, "error": f"HTTP {exc.response.status_code}", "url": url}
        except Exception as exc:
            return {"success": False, "error": str(exc), "url": url}

    async def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        """Search the web using DuckDuckGo.

        Returns: {success, results: [{title, url, snippet}], error}
        """
        try:
            from ddgs import DDGS

            ddgs = DDGS(timeout=10)
            raw = await asyncio.to_thread(ddgs.text, query, max_results=max_results)
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")[:300],
                }
                for r in (raw or [])
            ]
            return {"success": True, "results": results}
        except Exception as exc:
            return {"success": False, "results": [], "error": str(exc)}

    async def search_and_summarize(self, query: str, max_results: int = 3) -> dict[str, Any]:
        """Search, fetch top results, return combined content.

        Returns: {success, content, sources: [{title, url}], tokens}
        """
        search_result = await self.search(query, max_results=max_results + 2)
        if not search_result["success"]:
            return {
                "success": False,
                "content": "",
                "sources": [],
                "error": search_result.get("error"),
            }

        sources: list[dict[str, str]] = []
        contents: list[str] = []
        for item in search_result["results"][:max_results]:
            fetch_result = await self.fetch(item["url"])
            if fetch_result["success"]:
                sources.append({"title": item["title"], "url": item["url"]})
                contents.append(
                    f"## {item['title']}\nSource: {item['url']}\n\n"
                    f"{fetch_result['content'][:2000]}"
                )

        combined = "\n\n---\n\n".join(contents)
        token_est = max(1, int(len(combined.split()) * 1.3))

        return {
            "success": len(contents) > 0,
            "content": combined,
            "sources": sources,
            "tokens": token_est,
        }

    def _html_to_markdown(self, content: str) -> str:
        """Convert HTML to clean markdown for LLM consumption."""
        # Remove scripts and styles
        text = re.sub(
            r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(
            r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        # Convert headings
        text = re.sub(
            r"<h([1-6])[^>]*>(.*?)</h\1>",
            lambda m: f"\n{'#' * int(m.group(1))} {m.group(2).strip()}\n",
            text,
            flags=re.IGNORECASE,
        )
        # Convert links
        text = re.sub(
            r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            r"[\2](\1)",
            text,
            flags=re.IGNORECASE,
        )
        # Convert list items
        text = re.sub(
            r"<li[^>]*>(.*?)</li>",
            lambda m: f"\n- {m.group(1).strip()}",
            text,
            flags=re.IGNORECASE,
        )
        # Convert paragraphs
        text = re.sub(r"</(p|div|section|article)>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<(br|hr)\s*/?>", "\n", text, flags=re.IGNORECASE)
        # Strip remaining tags
        text = re.sub(r"<[^>]+>", "", text)
        # Decode entities
        text = html.unescape(text)
        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
