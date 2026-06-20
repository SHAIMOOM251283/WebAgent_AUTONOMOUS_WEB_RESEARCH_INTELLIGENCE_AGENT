"""
app/app.py
==========
Flask web application for WebAgent.
Serves the frontend and connects it to the ResearchAgent.

Routes:
  GET  /           — serves index.html
  POST /research   — streams the agent's research report back to the browser
  GET  /health     — checks both LLM providers are reachable

Streaming:
  /research uses Flask's stream_with_context to send the report
  token by token as the agent produces it, so the user sees
  output appearing in real time rather than waiting 30-60 seconds.

Usage
-----
    cd WebAgent
    python app/app.py
"""

import json
import logging
import sys
import os

from flask import Flask, render_template, request, Response, stream_with_context
from flask_cors import CORS
from langchain_core.messages import HumanMessage

# Allow imports from the project root regardless of where Flask is launched from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agent import ResearchAgent
from agent.llm import LanguageModelManager


class WebAgentApp:

    # ── Initialise all variables ───────────────────────────────
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.HOST      = "0.0.0.0"
        self.PORT      = 5000
        self.DEBUG     = True
        self.TEMPLATES = os.path.join(os.path.dirname(__file__), "templates")
        self.STATIC    = os.path.join(os.path.dirname(__file__), "static")

        self.app            = Flask(__name__, template_folder=self.TEMPLATES, static_folder=self.STATIC)
        self.app.secret_key = "webagent-dev-secret-key"
        CORS(self.app)

        self.agent       = ResearchAgent()
        self.llm_manager = LanguageModelManager()

        self._register_routes()

        self.logger.info("✅ WebAgentApp initialised on %s:%d", self.HOST, self.PORT)

    # ── Register all routes in one place ────────────────────────
    def _register_routes(self):
        """
        Register all URL rules.
        Acts as a table of contents for the application's API.
        """
        self.app.add_url_rule(
            "/",
            view_func=self.index,
            methods=["GET"],
        )
        self.app.add_url_rule(
            "/research",
            view_func=self.research,
            methods=["POST"],
        )
        self.app.add_url_rule(
            "/health",
            view_func=self.health,
            methods=["GET"],
        )

    # ── Serve the frontend ──────────────────────────────────────
    def index(self):
        """Serve the main frontend page."""
        return render_template("index.html")

    # ── Stream the research report ───────────────────────────────
    def research(self):
        """
        Receive a research goal from the frontend and stream the report back.

        Expects JSON body: { "goal": "your research goal here" }

        Streams Server-Sent Events (SSE) so the frontend receives the report
        progressively rather than waiting for the full response.

        SSE format:
          data: {"type": "chunk",  "content": "...text..."}
          data: {"type": "done",   "content": ""}
          data: {"type": "error",  "content": "...message..."}
        """
        data = request.get_json(silent=True)

        if not data or not data.get("goal", "").strip():
            return Response(
                json.dumps({"error": "No research goal provided."}),
                status=400,
                mimetype="application/json",
            )

        goal = data["goal"].strip()
        self.logger.info("📋 Received research goal: '%s'", goal[:80])

        def generate():
            try:
                # Prepare a fresh session
                self.agent.memory.clear()
                self.logger.info("🎯 Streaming research session started.")

                initial_state = {
                    "messages": [HumanMessage(content=goal)]
                }

                config = {"recursion_limit": self.agent.MAX_ITERATIONS}

                # Stream chunks from the LangGraph agent
                for chunk in self.agent.graph.stream(initial_state, config=config):
                    for node_name, node_output in chunk.items():
                        messages = node_output.get("messages", [])
                        if messages:
                            last    = messages[-1]
                            content = last.content

                            # Only stream non-empty text content from the agent node
                            if content and isinstance(content, str) and node_name == "agent":
                            #if content and isinstance(content, str) and node_name in ("agent", "tools"):
                                sse_data = json.dumps({
                                    "type"   : "chunk",
                                    "content": content,
                                    "node"   : node_name,
                                })
                                yield f"data: {sse_data}\n\n"

                # Signal completion
                yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
                self.logger.info("✅ Streaming complete for goal: '%s'", goal[:80])

            except Exception as exc:
                self.logger.error("❌ Streaming error: %s", exc)
                yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control"    : "no-cache",
                "X-Accel-Buffering": "no",   # disables Nginx buffering if deployed
            },
        )

    # ── Health check for both LLM providers ─────────────────────
    def health(self):
        """
        Ping both Groq and Gemini and return their status.
        Useful for confirming API keys are valid before a research session.
        """
        self.logger.info("🏥 Health check requested.")

        health_data = self.llm_manager.check_llm_health()

        overall = "ok" if all(
            v.get("status") == "ok" for v in health_data.values()
        ) else "degraded"

        return Response(
            json.dumps({
                "status"   : overall,
                "providers": health_data,
            }),
            status=200,
            mimetype="application/json",
        )

    # ── Run the Flask development server ──────────────────────────
    def run(self):
        logging.basicConfig(level=logging.INFO)
        self.app.run(
            host=self.HOST,
            port=self.PORT,
            debug=self.DEBUG,
            threaded=True,   # handles concurrent requests without blocking
        )


if __name__ == "__main__":
    WebAgentApp().run()
