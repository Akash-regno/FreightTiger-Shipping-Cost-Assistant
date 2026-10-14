"""
LLM Assistant module powered by Groq.
Provides interactive natural-language Q&A and explanation synthesis
grounded strictly in the shipment records, weekly costs, and context notes.

Includes complete token and cost tracking as requested in the case study brief.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"

# Pricing per million tokens (Groq free tier / public rate)
INPUT_PRICE_PER_M = 0.20
OUTPUT_PRICE_PER_M = 0.20


class GroqCostTracker:
    def __init__(self):
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        self.total_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_cost_usd(self) -> float:
        cost_in = (self.total_prompt_tokens / 1_000_000) * INPUT_PRICE_PER_M
        cost_out = (self.total_completion_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
        return cost_in + cost_out

    def get_summary(self) -> Dict[str, Any]:
        return {
            "calls": self.total_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.total_cost_usd, 6),
        }


# Global singleton tracker
tracker = GroqCostTracker()


def get_groq_client():
    """Return Groq client if key is configured, else None."""
    api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
    if not api_key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except Exception:
        return None


def ask_assistant(
    query: str,
    output_summary_text: str,
    model: str = DEFAULT_GROQ_MODEL
) -> Dict[str, Any]:
    """
    Ask the AI assistant a question about shipping cost anomalies and context notes.
    Grounded strictly in the analyzed data to prevent hallucination.
    """
    client = get_groq_client()
    if not client:
        return {
            "answer": "Groq API key not configured. Please check your .env file.",
            "tracker": tracker.get_summary()
        }

    system_prompt = (
        "You are the Freight Tiger Smart Freight Cost Assistant. "
        "Your task is to answer user inquiries about shipping costs, weekly cost anomalies, "
        "peer route comparisons, and real-world justifications.\n\n"
        "STRICT GUARDRAILS:\n"
        "1. Ground all answers strictly in the provided data summary and context notes.\n"
        "2. If a cost increase has an explanation (e.g. flood, festival surcharge), cite the exact note ID and dates.\n"
        "3. If a cost increase has NO explanation in the notes, state clearly that it is unexplained and flagged for human review.\n"
        "4. Never invent or speculate on reasons not found in the context notes.\n"
        "5. Be concise, direct, professional, and explain in plain English."
    )

    user_content = (
        f"CONTEXT DATA SUMMARY:\n{output_summary_text}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        "Provide a clear, accurate, plain-English response grounded in the context above."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,  # Strict determinism
            max_tokens=400,
        )

        answer = response.choices[0].message.content
        if hasattr(response, "usage") and response.usage:
            tracker.add_usage(
                response.usage.prompt_tokens or 0,
                response.usage.completion_tokens or 0,
            )

        return {
            "answer": answer,
            "tracker": tracker.get_summary()
        }
    except Exception as e:
        return {
            "answer": f"Error calling Groq API: {str(e)}",
            "tracker": tracker.get_summary()
        }
