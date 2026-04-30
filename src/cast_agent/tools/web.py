"""Web search wrapper using ddgs (formerly duckduckgo_search)."""


class WebSearchClient:
    """Search the web via DuckDuckGo (no API key required)."""

    def __init__(self):
        self._ddgs = None

    def _get_client(self):
        if self._ddgs is None:
            import warnings

            try:
                from ddgs import DDGS
            except ImportError:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    from duckduckgo_search import DDGS
            self._ddgs = DDGS()
        return self._ddgs

    def search(self, query: str, max_results: int = 5) -> str:
        try:
            client = self._get_client()
            results = list(client.text(query, max_results=max_results))
        except Exception as e:
            return f"Web search error: {e}"

        if not results:
            return f"No web results found for: {query}"

        parts = [f"{len(results)} result(s) for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            parts.append(f"[{i}] {title}\n    {body}\n    URL: {href}")

        return "\n\n".join(parts)
