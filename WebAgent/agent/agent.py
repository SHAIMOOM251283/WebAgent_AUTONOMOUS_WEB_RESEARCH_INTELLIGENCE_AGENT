"""
agent/agent.py
==============
Orchestrates the WebAgent research loop using LangGraph's create_react_agent.

The agent receives a research goal, autonomously decides which tools to use,
searches the web, fetches pages, stores findings, and generates a final
structured intelligence report — without the user specifying any steps.

Usage
-----
    from agent.agent import ResearchAgent

    agent = ResearchAgent()
    report = agent.run("What are the latest developments in LangGraph?")
    print(report)
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent  

from agent.llm import LanguageModelManager
from agent.memory import ResearchMemory
from agent.tools import AgentTools


class ResearchAgent:

    # ── Initialise all variables ───────────────────────────────
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.MAX_ITERATIONS = 10    # safety limit on the agent loop

        self.SYSTEM_PROMPT = """
You are an autonomous web research agent. Your job is to thoroughly research
a given goal by using your tools in this order:

1. Use web_search to find relevant pages on the topic.
2. Use fetch_page to extract content from the most promising URLs.
3. Use store_finding to save every important piece of information you find,
   along with its source URL.
4. If you do not yet have enough information, use web_search again with a
   refined or different query and repeat steps 2-3.
5. Once you have gathered sufficient findings (aim for at least 3-5 stored
   findings), use generate_report to produce the final structured report.

Important rules:
- Always store findings before generating the report.
- Never generate the report with zero stored findings.
- Be selective: only fetch pages that are directly relevant to the goal.
- Be thorough: cover multiple angles of the research goal.
- The final output must come from generate_report, not from your own words.
""".strip()

        self.llm_manager = LanguageModelManager()
        self.memory       = ResearchMemory()
        self.tools_manager = AgentTools()

        self.tools = self.tools_manager.get_tools()
        self.llm   = self.llm_manager.get_llm("groq")
        #self.llm   = self.llm_manager.get_llm("gemini")
        self.graph = self._build_graph()

    # ── Bind tools to the LLM and build the LangGraph ───────────
    def _build_graph(self):
        """
        Bind the four tools to the LLM and create the ReAct agent graph.
        create_react_agent handles the agent node, tool node, and
        conditional edges automatically.
        """
        llm_with_tools = self.llm.bind_tools(self.tools)

        graph = create_react_agent(
            model=llm_with_tools,
            tools=self.tools,
            prompt=self.SYSTEM_PROMPT,
        )

        self.logger.info("✅ ResearchAgent graph built with %d tools.", len(self.tools))
        return graph

    # ── Clear memory before each new research session ───────────
    def _prepare_session(self, goal: str) -> None:
        """
        Clear any findings from a previous session and log the new goal.
        Called at the start of every run() invocation.
        """
        self.memory.clear()
        self.logger.info("🎯 New research session started.")
        self.logger.info("📋 Goal: %s", goal)

    # ── Stream the agent loop and capture the final message ──────
    def _execute(self, goal: str) -> Optional[str]:
        """
        Feed the goal into the LangGraph agent and stream the response.
        Returns the content of the final AIMessage, or None on failure.
        """
        initial_state = {
            "messages": [HumanMessage(content=goal)],
        }

        config = {
            "recursion_limit": self.MAX_ITERATIONS,
        }

        final_message = None

        try:
            for chunk in self.graph.stream(initial_state, config=config):
                # Each chunk is a dict: {"node_name": {"messages": [...]}}
                for node_name, node_output in chunk.items():
                    messages = node_output.get("messages", [])
                    if messages:
                        last = messages[-1]
                        self.logger.debug(
                            "📨 [%s] %s: %s…",
                            node_name,
                            type(last).__name__,
                            str(last.content)[:80],
                        )
                        final_message = last

        except Exception as exc:
            self.logger.error("❌ Agent execution failed: %s", exc)
            return None

        if final_message is None:
            self.logger.error("❌ Agent produced no output.")
            return None

        return final_message.content

    # ── Main public method — run a full research session ─────────
    def run(self, goal: str) -> str:
        """
        Run a complete research session for the given goal.

        Args:
            goal: The research goal entered by the user.

        Returns:
            A structured intelligence report as a markdown string,
            or an error message if the session failed.
        """
        self._prepare_session(goal)

        result = self._execute(goal)

        if result is None:
            return (
                "The research agent encountered an error and could not "
                "complete the session. Please try again."
            )

        self.logger.info("✅ Research session complete.")
        return result

    # ── Self-test ─────────────────────────────────────────────────
    def run_test(self) -> None:
        logging.basicConfig(level=logging.INFO)

        print("\n🤖 ResearchAgent — Self Test\n" + "─" * 40)

        goal = "What is LangGraph and how is it used to build AI agents?"

        print(f"\n🎯 Research Goal: {goal}\n")
        print("⏳ Running agent loop… (this may take 30-60 seconds)\n")

        report = self.run(goal)

        print("\n" + "═" * 60)
        print("📊 FINAL REPORT")
        print("═" * 60)
        print(report)
        print("═" * 60)
        print("\n✅ ResearchAgent self-test complete.")


if __name__ == "__main__":
    ResearchAgent().run_test()
