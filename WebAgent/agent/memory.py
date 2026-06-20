"""
agent/memory.py
===============
Manages persistent storage and retrieval of research findings using ChromaDB.
Embeddings are generated locally using sentence-transformers (CPU-friendly).

Usage
-----
    from agent.memory import ResearchMemory

    memory = ResearchMemory()
    memory.store_finding("AI is transforming healthcare.", "https://example.com")
    results = memory.retrieve_findings("AI in medicine", n_results=3)
    memory.clear()
"""

import logging
import uuid
from typing import List, Dict

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class ResearchMemory:

    # ── Initialise all variables ───────────────────────────────
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.DB_PATH         = "./data/chroma_db"
        self.COLLECTION_NAME = "webagent_findings"
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # lightweight, CPU-friendly
        self.N_RESULTS       = 5                    # default retrieval count

        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self.EMBEDDING_MODEL
        )

        self.client     = chromadb.PersistentClient(path=self.DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_function,
        )

        self.logger.info(
            "✅ ResearchMemory initialised — collection '%s' has %d findings.",
            self.COLLECTION_NAME,
            self.collection.count(),
        )

    # ── Store a single research finding ─────────────────────────
    def store_finding(self, content: str, source: str) -> str:
        """
        Embed and store a research finding.

        Args:
            content : The extracted text from a web page.
            source  : The URL the content came from.

        Returns:
            The unique ID assigned to this finding.
        """
        finding_id = str(uuid.uuid4())

        self.collection.add(
            ids=[finding_id],
            documents=[content],
            metadatas=[{"source": source}],
        )

        self.logger.info(
            "💾 Stored finding from '%s' (id: %s…)", source, finding_id[:8]
        )

        return finding_id

    # ── Retrieve relevant findings by semantic similarity ────────
    def retrieve_findings(
        self, query: str, n_results: int = None
    ) -> List[Dict]:
        """
        Retrieve the most semantically relevant findings for a query.

        Args:
            query    : The search query or research goal.
            n_results: Number of results to return (defaults to self.N_RESULTS).

        Returns:
            A list of dicts, each with 'content' and 'source' keys.
        """
        n = n_results or self.N_RESULTS

        total = self.collection.count()
        if total == 0:
            self.logger.warning("⚠️  No findings in memory yet.")
            return []

        # ChromaDB raises an error if n_results exceeds total documents
        n = min(n, total)

        results = self.collection.query(
            query_texts=[query],
            n_results=n,
        )

        findings = []
        for content, metadata in zip(
            results["documents"][0], results["metadatas"][0]
        ):
            findings.append({
                "content": content,
                "source" : metadata.get("source", "unknown"),
            })

        self.logger.info(
            "🔍 Retrieved %d findings for query: '%s'", len(findings), query[:60]
        )

        return findings

    # ── Report how many findings are currently stored ────────────
    def count(self) -> int:
        """Return the total number of findings currently in the collection."""
        return self.collection.count()

    # ── Clear all findings from the collection ──────────────────
    def clear(self) -> None:
        """
        Delete all findings from the collection.
        Called at the start of each new research session so old findings
        do not pollute a new research goal.
        """
        total = self.collection.count()

        if total == 0:
            self.logger.info("ℹ️  Collection already empty — nothing to clear.")
            return

        # Fetch all IDs and delete them
        all_ids = self.collection.get()["ids"]
        self.collection.delete(ids=all_ids)

        self.logger.info("🗑️  Cleared %d findings from memory.", total)

    # ── Self-test ─────────────────────────────────────────────────
    def run(self) -> None:
        logging.basicConfig(level=logging.INFO)

        print("\n🧠 ResearchMemory — Self Test\n" + "─" * 40)

        # Clear any leftover findings from previous runs
        self.clear()

        # Store three test findings
        self.store_finding(
            content="LangGraph is a library for building stateful multi-agent applications.",
            source="https://example.com/langgraph",
        )
        self.store_finding(
            content="Groq provides ultra-fast LLM inference on custom hardware.",
            source="https://example.com/groq",
        )
        self.store_finding(
            content="ChromaDB is an open-source vector database for AI applications.",
            source="https://example.com/chromadb",
        )

        print(f"\n📦 Total findings stored: {self.count()}")

        # Retrieve findings relevant to a query
        print("\n🔍 Retrieving findings for query: 'fast language model inference'")
        results = self.retrieve_findings("fast language model inference", n_results=2)
        for i, r in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Source  : {r['source']}")
            print(f"    Content : {r['content']}")

        # Clear and confirm
        self.clear()
        print(f"\n📦 Findings after clear: {self.count()}")
        print("\n✅ ResearchMemory self-test complete.")


if __name__ == "__main__":
    ResearchMemory().run()
