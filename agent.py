"""
Eco Lifestyle Agent — LLM-powered agent that uses RAG context to answer
eco-lifestyle questions with actionable, friendly advice.

Supports:
  - OpenAI chat models (gpt-3.5-turbo, gpt-4o, …)
  - Ollama local models (llama3, mistral, gemma, …)
  - A rules-based fallback when no LLM API key is configured
"""
from __future__ import annotations

from typing import List, Optional

import config
from rag_engine import get_pipeline, Document

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are EcoAgent 🌿, a friendly and knowledgeable AI assistant dedicated to helping people live a more sustainable, eco-friendly lifestyle.

Your role is to:
- Provide **clear, practical, and actionable** advice on sustainable living
- Draw on the retrieved knowledge context provided with each question
- Recommend **specific products, schemes, and government grants** where relevant
- Highlight the **environmental impact** of suggested actions (CO₂ saved, water saved, etc.)
- Use an **encouraging, positive tone** — sustainability should feel accessible, not overwhelming
- Structure responses with **short paragraphs or bullet points** for readability
- When appropriate, mention relevant **government schemes or financial incentives**

Guidelines:
- Always base your answers on the provided context. If the context does not contain relevant information, say so honestly.
- Do not invent statistics, product names, or scheme names.
- Keep responses concise (200–400 words) unless more detail is genuinely required.
- Use emojis sparingly to add warmth (1–3 per response maximum).
- End each response with one **actionable "Quick Win"** the user can do today.
"""

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(question: str, context: str, chat_history: Optional[List[dict]] = None) -> List[dict]:
    """Build the messages list for the chat completion API."""
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include recent chat history (last 6 exchanges) for multi-turn context
    if chat_history:
        for turn in chat_history[-6:]:
            messages.append(turn)

    # Inject retrieved context
    user_content = f"""Context from eco-lifestyle knowledge base:
---
{context}
---

User question: {question}

Please provide a helpful, practical, and accurate answer based on the context above."""

    messages.append({"role": "user", "content": user_content})
    return messages


# ── LLM clients ───────────────────────────────────────────────────────────────

def _call_openai(messages: List[dict]) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def _call_ollama(messages: List[dict]) -> str:
    import httpx
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    resp = httpx.post(
        f"{config.OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _call_fallback(question: str, context: str) -> str:
    """Rule-based fallback when no LLM is configured. Returns retrieved context summary."""
    lines = []
    lines.append("🌿 **Eco Lifestyle Agent** (Context-only mode — no LLM configured)")
    lines.append("")
    lines.append(f"**Your question:** {question}")
    lines.append("")
    lines.append("**Relevant information from the knowledge base:**")
    lines.append("")
    # Pull first 600 chars of context
    snippet = context[:1200].strip()
    if len(context) > 1200:
        snippet += "…"
    lines.append(snippet)
    lines.append("")
    lines.append("_To get full AI-powered answers, add your OpenAI API key to the `.env` file "
                 "or run a local Ollama model._")
    return "\n".join(lines)


# ── Agent class ───────────────────────────────────────────────────────────────

class EcoAgent:
    """
    The main agent.  Usage::

        agent = EcoAgent()
        answer = agent.chat("How can I reduce plastic use at home?")
    """

    def __init__(self):
        self._pipeline = get_pipeline()
        self._history: List[dict] = []

    # ── public API ─────────────────────────────────────────────────────────────
    def chat(self, question: str, top_k: int | None = None) -> dict:
        """
        Process a question and return a response dict::

            {
              "answer":  str,
              "sources": List[str],
              "context_chunks": int,
            }
        """
        # 1. Retrieve relevant context
        docs: List[Document] = self._pipeline.retrieve(question, top_k=top_k)
        context = self._pipeline.retrieve_context(question, top_k=top_k)
        sources = sorted({doc.source for doc in docs})

        # 2. Build messages
        messages = build_prompt(question, context, chat_history=self._history)

        # 3. Call LLM
        answer = self._generate(question, context, messages)

        # 4. Update history
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": sources,
            "context_chunks": len(docs),
        }

    def reset_history(self) -> None:
        """Clear the conversation history."""
        self._history = []

    # ── private ────────────────────────────────────────────────────────────────
    def _generate(self, question: str, context: str, messages: List[dict]) -> str:
        provider = config.LLM_PROVIDER.lower()

        if provider == "openai" and config.OPENAI_API_KEY:
            try:
                return _call_openai(messages)
            except Exception as exc:
                return f"⚠️ OpenAI error: {exc}\n\n" + _call_fallback(question, context)

        if provider == "ollama":
            try:
                return _call_ollama(messages)
            except Exception as exc:
                return f"⚠️ Ollama error: {exc}\n\n" + _call_fallback(question, context)

        return _call_fallback(question, context)


# ── Singleton ─────────────────────────────────────────────────────────────────
_agent: EcoAgent | None = None


def get_agent() -> EcoAgent:
    global _agent
    if _agent is None:
        _agent = EcoAgent()
    return _agent
