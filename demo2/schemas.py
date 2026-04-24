"""
schemas.py  ─  Pydantic v2 output models
=========================================
This is where we define WHAT the LLM must return.
LangChain passes this schema to the LLM so it generates
valid, structured JSON that we can validate and use directly.

WHY PYDANTIC OUTPUT?
  Normal LLM: returns a raw string → you have to parse it yourself
              and it may be missing fields or have wrong types.

  Pydantic output: LLM is forced to return a specific JSON structure.
                   Pydantic validates every field, type, and constraint.
                   Your app always gets a clean Python object, not a string.

PYDANTIC v2 NOTES:
  - Use  model_validator  for cross-field validation
  - Use  field_validator  for single-field validation
  - Use  Field(...)  for metadata like description, examples, constraints
  - Literal["a","b"]  restricts to allowed values only
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class KeyFact(BaseModel):
    """One important fact extracted from the documents."""

    fact: str = Field(
        description="A single, concise medical fact from the source document.",
        min_length=10,
        max_length=300,
    )
    topic: str = Field(
        description="The medical topic this fact belongs to (e.g. 'diabetes', 'renal failure', 'obesity').",
    )

    # field_validator: cleans the fact text before storing
    @field_validator("fact")
    @classmethod
    def capitalize_fact(cls, v: str) -> str:
        """Ensure every fact starts with a capital letter."""
        return v.strip().capitalize()


class MedicalAnswer(BaseModel):
    """
    Structured answer to a medical question.
    The LLM MUST fill every field — no free-form text blob.
    """

    # --- Core answer ---
    answer: str = Field(
        description="A clear, complete answer to the question based only on the provided context.",
        min_length=20,
    )

    # --- Conditions detected in the question/answer ---
    conditions_mentioned: list[str] = Field(
        description="List of medical conditions referenced (e.g. ['Type 2 Diabetes', 'Chronic Kidney Disease']).",
        default_factory=list,
    )

    # --- Key facts pulled from the documents ---
    key_facts: list[KeyFact] = Field(
        description="2–5 important facts from the source documents relevant to this question.",
        min_length=1,
        max_length=5,
    )

    # --- Confidence: how well do the docs cover this question? ---
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "high   = answer is clearly supported by the documents. "
            "medium = partially supported. "
            "low    = answer is inferred or documents lack enough detail."
        )
    )

    # --- Follow-up suggestions ---
    follow_up_questions: list[str] = Field(
        description="2–3 related follow-up questions the user might want to ask next.",
        min_length=2,
        max_length=3,
    )

    # --- Always present a medical disclaimer ---
    disclaimer: str = Field(
        description="A short disclaimer reminding the user to consult a healthcare professional.",
        default="This information is for educational purposes only. Always consult a qualified healthcare provider.",
    )

    @model_validator(mode="after")
    def check_low_confidence_answer(self) -> "MedicalAnswer":
        """
        If confidence is 'low', the answer must contain a warning phrase.
        This prevents the model from giving a confident-sounding answer
        when the source documents don't actually support it well.
        """
        if self.confidence == "low":
            warning_phrases = ["not clearly covered", "limited information", "may not be fully"]
            has_warning = any(phrase in self.answer.lower() for phrase in warning_phrases)
            if not has_warning:
                # Append a warning instead of raising an error
                # (so we don't crash the app — we just fix the output)
                self.answer += " (Note: the source documents have limited information on this topic.)"
        return self

    # ── Convert to a display-friendly dict ───────────────────────────────────
    def to_display(self) -> dict:
        """Returns a clean dict for rendering in Streamlit."""
        return {
            "answer": self.answer,
            "conditions": self.conditions_mentioned,
            "key_facts": [{"fact": kf.fact, "topic": kf.topic} for kf in self.key_facts],
            "confidence": self.confidence,
            "follow_up": self.follow_up_questions,
            "disclaimer": self.disclaimer,
        }