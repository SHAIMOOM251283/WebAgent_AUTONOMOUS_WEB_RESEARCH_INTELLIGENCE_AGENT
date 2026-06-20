"""
agent/tools.py
==============
Defines the four tools available to the WebAgent:
  1. web_search(query)            — DuckDuckGo search, returns URLs and snippets
  2. fetch_page(url)              — Playwright fetch, returns clean text
  3. store_finding(content, source) — stores a finding in ChromaDB
  4. generate_report(goal)        — retrieves all findings and generates a structured report

Usage
-----
    from agent.tools import AgentTools

    tools = AgentTools()
    tool_list = tools.get_tools()  # pass this list to LangGraph
"""

import logging
from typing import Optional

from ddgs import DDGS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import StructuredTool

from agent.llm import LanguageModelManager
from agent.memory import ResearchMemory
from scraper.playwright_fetch import PageFetcher


class AgentTools:

    # ── Initialise all variables ───────────────────────────────
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.MAX_SEARCH_RESULTS = 5     # DuckDuckGo results per query
        self.MAX_PAGES_TO_FETCH = 3     # Playwright fetches per research loop

        self.llm_manager = LanguageModelManager()
        self.memory       = ResearchMemory()
        self.fetcher      = PageFetcher()

        self.REPORT_SYSTEM_PROMPT = """
You are a professional research analyst. You will be given a research goal and
a collection of findings gathered from the web. Your task is to produce a
structured intelligence report with exactly these four sections:

## Summary
A concise 2-3 sentence overview of what the research found.

## Key Findings
A numbered list of the most important insights from the research.

## Sources
A numbered list of all source URLs referenced in the findings.

## Gaps
A brief description of what the research could not conclusively answer
and what further investigation might be needed.

Write clearly and objectively. Do not add any sections beyond the four above.
""".strip()

    # ── Search the web using DuckDuckGo ─────────────────────────
    def web_search(self, query: str) -> str:
        """
        Search DuckDuckGo for a query and return URLs with snippets.

        Args:
            query: The search query string.

        Returns:
            A formatted string of results, or an error message.
        """
        self.logger.info("🔎 Searching: '%s'", query)

        try:
            results = []
            with DDGS() as ddgs:
                for result in ddgs.text(query, max_results=self.MAX_SEARCH_RESULTS):
                    results.append(result)

            if not results:
                self.logger.warning("⚠️  No results found for query: '%s'", query)
                return "No results found."

            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"{i}. {r.get('title', 'No title')}\n"
                    f"   URL: {r.get('href', '')}\n"
                    f"   Snippet: {r.get('body', '')}"
                )

            self.logger.info("✅ Found %d results for '%s'.", len(results), query)
            return "\n\n".join(formatted)

        except Exception as exc:
            self.logger.error("❌ Search failed for '%s': %s", query, exc)
            return f"Search failed: {exc}"

    # ── Fetch and extract text from a web page ──────────────────
    def fetch_page(self, url: str) -> str:
        """
        Fetch a URL using Playwright and return clean extracted text.

        Args:
            url: The full URL to fetch.

        Returns:
            Extracted text, or an error message if the fetch failed.
        """
        self.logger.info("🌐 Fetching page: '%s'", url)

        text = self.fetcher.fetch(url)

        if text is None:
            self.logger.warning("⚠️  fetch_page returned None for '%s'.", url)
            return f"Could not fetch page: {url}"

        return text

    # ── Store a research finding in ChromaDB ────────────────────
    def store_finding(self, content: str, source: str) -> str:
        """
        Store a research finding in ChromaDB as a vector embedding.

        Args:
            content: The text content of the finding.
            source : The URL the content came from.

        Returns:
            A confirmation message with the assigned finding ID.
        """
        self.logger.info("💾 Storing finding from '%s'.", source)

        finding_id = self.memory.store_finding(content=content, source=source)
        total      = self.memory.count()

        return (
            f"Finding stored successfully. "
            f"ID: {finding_id[:8]}… | Total findings: {total}"
        )

    # ── Generate the final structured intelligence report ────────
    def generate_report(self, goal: str) -> str:
        """
        Retrieve all stored findings and ask the LLM to generate
        a structured intelligence report.

        Args:
            goal: The original research goal entered by the user.

        Returns:
            A structured markdown report as a string.
        """
        self.logger.info("📝 Generating report for goal: '%s'", goal[:60])

        findings = self.memory.retrieve_findings(query=goal, n_results=20)

        if not findings:
            return (
                "No findings were stored during this research session. "
                "The agent was unable to gather sufficient information."
            )

        # Format findings into a single context block for the LLM
        findings_text = ""
        for i, f in enumerate(findings, 1):
            findings_text += (
                f"\n--- Finding {i} ---\n"
                f"Source: {f['source']}\n"
                f"Content: {f['content']}\n"
            )

        human_message = (
            f"Research Goal: {goal}\n\n"
            f"Findings:\n{findings_text}\n\n"
            "Please generate the structured intelligence report now."
        )

        messages = [
            SystemMessage(content=self.REPORT_SYSTEM_PROMPT),
            HumanMessage(content=human_message),
        ]

        response = self.llm_manager.invoke_with_fallback(messages)

        self.logger.info("✅ Report generated successfully.")
        return response.content

    # ── Wrap all tools for LangGraph consumption ────────────────
    def get_tools(self) -> list:
        """
        Return all four tools as LangChain StructuredTool objects.
        Pass the returned list directly to the LangGraph agent.
        """
        web_search_tool = StructuredTool.from_function(
            func=self.web_search,
            name="web_search",
            description=(
                "Search the web using DuckDuckGo. "
                "Input: a search query string. "
                "Output: a list of URLs with titles and snippets."
            ),
        )

        fetch_page_tool = StructuredTool.from_function(
            func=self.fetch_page,
            name="fetch_page",
            description=(
                "Fetch a web page and extract its text content. "
                "Input: a full URL. "
                "Output: clean extracted text from the page."
            ),
        )

        store_finding_tool = StructuredTool.from_function(
            func=self.store_finding,
            name="store_finding",
            description=(
                "Store an important research finding in memory. "
                "Input: content (the text finding) and source (the URL). "
                "Output: confirmation message."
            ),
        )

        generate_report_tool = StructuredTool.from_function(
            func=self.generate_report,
            name="generate_report",
            description=(
                "Generate a structured intelligence report from all stored findings. "
                "Input: the original research goal. "
                "Output: a structured markdown report with Summary, "
                "Key Findings, Sources, and Gaps sections."
            ),
        )

        return [
            web_search_tool,
            fetch_page_tool,
            store_finding_tool,
            generate_report_tool,
        ]

    # ── Self-test ─────────────────────────────────────────────────
    def run(self) -> None:
        logging.basicConfig(level=logging.INFO)

        print("\n🛠️  AgentTools — Self Test\n" + "─" * 40)

        # Clear memory before testing
        self.memory.clear()

        # Test web_search
        print("\n1️⃣  Testing web_search…")
        search_result = self.web_search("LangGraph autonomous agents 2024")
        print(search_result[:400], "…")

        # Test fetch_page
        print("\n2️⃣  Testing fetch_page…")
        page_text = self.fetch_page("https://en.wikipedia.org/wiki/LangChain")
        print(f"Fetched {len(page_text)} characters.")
        print(page_text[:300], "…")

        # Test store_finding
        print("\n3️⃣  Testing store_finding…")
        confirmation = self.store_finding(
            content=page_text[:500],
            source="https://en.wikipedia.org/wiki/LangChain",
        )
        print(confirmation)

        # Test generate_report
        print("\n4️⃣  Testing generate_report…")
        report = self.generate_report("What is LangChain?")
        print(report)

        # Confirm tool list
        print("\n5️⃣  Confirming get_tools() returns 4 tools…")
        tools = self.get_tools()
        print(f"Tools available: {[t.name for t in tools]}")

        print("\n✅ AgentTools self-test complete.")


if __name__ == "__main__":
    AgentTools().run()
