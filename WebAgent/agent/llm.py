"""
agent/llm.py — class-based style
================================
Primary  : Groq  → llama-3.1-8b-instant
Fallback : Gemini → gemini-1.5-flash
"""

import json
import logging
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import (
    Retrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from groq import RateLimitError as GroqRateLimitError
from groq import APIStatusError as GroqAPIStatusError


class LanguageModelManager:

    # ── Initialise all variables ───────────────────────────────
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.GROQ_API_KEY   = "Groq_API_Key"
        self.GEMINI_API_KEY = "Gemini_API_Key"

        self.GROQ_MODEL   = "llama-3.1-8b-instant"
        self.GEMINI_MODEL = "gemini-2.5-flash-lite"

        self.TEMPERATURE = 0.3
        self.MAX_TOKENS  = 2048
        self.MAX_RETRIES = 2

        self._groq_llm   = None
        self._gemini_llm = None

    # ── Build the Groq LLM ──────────────────────────────────────
    def _build_groq_llm(self) -> ChatGroq:
        return ChatGroq(
            model=self.GROQ_MODEL,
            api_key=self.GROQ_API_KEY,
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
        )

    # ── Build the Gemini LLM ────────────────────────────────────
    def _build_gemini_llm(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.GEMINI_MODEL,
            google_api_key=self.GEMINI_API_KEY,
            temperature=self.TEMPERATURE,
            max_output_tokens=self.MAX_TOKENS,
            convert_system_message_to_human=True,
        )

    # ── Return a cached LLM instance ────────────────────────────
    def get_llm(self, provider: str = "groq") -> ChatGroq | ChatGoogleGenerativeAI:
        if provider == "groq":
            if self._groq_llm is None:
                self._groq_llm = self._build_groq_llm()
                self.logger.info("✅ Groq LLM initialised (%s)", self.GROQ_MODEL)
            return self._groq_llm

        if provider == "gemini":
            if self._gemini_llm is None:
                self._gemini_llm = self._build_gemini_llm()
                self.logger.info("✅ Gemini LLM initialised (%s)", self.GEMINI_MODEL)
            return self._gemini_llm

        raise ValueError(f"Unknown provider '{provider}'. Choose 'groq' or 'gemini'.")

    # ── Call Groq with retry logic (no decorator needed) ────────
    def _call_groq(self, messages: List[BaseMessage]) -> BaseMessage:
        retryer = Retrying(
            retry=retry_if_exception_type((GroqRateLimitError, GroqAPIStatusError)),
            stop=stop_after_attempt(self.MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        return retryer(self.get_llm("groq").invoke, messages)

    # ── Main entry point with fallback ──────────────────────────
    def invoke_with_fallback(self, messages: List[BaseMessage]) -> BaseMessage:
        try:
            response = self._call_groq(messages)
            self.logger.debug("📡 Response from Groq (%s)", self.GROQ_MODEL)
            return response

        except Exception as groq_err:
            self.logger.warning(
                "⚠️  Unexpected Groq error (%s: %s). Switching to Gemini…",
                type(groq_err).__name__,
                groq_err,
            )

        try:
            gemini = self.get_llm("gemini")
            response = gemini.invoke(messages)
            self.logger.info("📡 Response from Gemini fallback (%s)", self.GEMINI_MODEL)
            return response

        except Exception as gemini_err:
            self.logger.error("❌ Gemini also failed: %s", gemini_err)
            raise RuntimeError(
                f"Both LLM providers failed.\n"
                f"  Groq error  : see logs above\n"
                f"  Gemini error: {gemini_err}\n\n"
                "Check your API keys and your network connection."
            ) from gemini_err

    # ── Health check for both providers ─────────────────────────
    def check_llm_health(self) -> dict:
        results = {}
        ping = [HumanMessage(content="Reply with the single word: OK")]

        for provider in ("groq", "gemini"):
            try:
                llm = self.get_llm(provider)
                reply = llm.invoke(ping)
                results[provider] = {
                    "status": "ok",
                    "model": self.GROQ_MODEL if provider == "groq" else self.GEMINI_MODEL,
                    "response_preview": reply.content[:40],
                }
            except Exception as exc:
                results[provider] = {
                    "status": "error",
                    "error": str(exc),
                }

        return results

    # ── Run the self-test ─────────────────────────────────────────
    def run(self):
        logging.basicConfig(level=logging.INFO)

        print("\n🔍 WebAgent — LLM Health Check\n" + "─" * 40)
        health = self.check_llm_health()
        print(json.dumps(health, indent=2))

        print("\n🚀 Testing invoke_with_fallback…")
        test_messages = [
            SystemMessage(content="You are a concise research assistant."),
            HumanMessage(content="In one sentence, what is LangGraph?"),
        ]
        result = self.invoke_with_fallback(test_messages)
        print(f"\n✅ Response:\n{result.content}")


if __name__ == "__main__":
    LanguageModelManager().run()
