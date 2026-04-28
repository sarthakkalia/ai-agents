from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

class KeyFact(BaseModel):
    fact: str = Field(
        description="A single, concise medical fact from the source document.",
        min_length=10,
        max_length=300,
    )
    topic: str = Field(
        description="The medical topic this fact belongs to (e.g. 'diabetes', 'renal failure', 'obesity').",
    )

    @field_validator("fact")
    @classmethod
    def capitalize_fact(cls, v: str) -> str:
        return v.strip().capitalize()


class MedicalAnswer(BaseModel):
    answer: str = Field(
        description="A clear, complete answer to the question based only on the provided context.",
        min_length=20,
    )
    conditions_mentioned: list[str] = Field(
        description="List of medical conditions referenced (e.g. ['Type 2 Diabetes', 'Chronic Kidney Disease']).",
        default_factory=list,
    )
    key_facts: list[KeyFact] = Field(
        description="2–5 important facts from the source documents relevant to this question.",
        min_length=1,
        max_length=5,
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "high   = answer is clearly supported by the documents. "
            "medium = partially supported. "
            "low    = answer is inferred or documents lack enough detail."
        )
    )
    follow_up_questions: list[str] = Field(
        description="2–3 related follow-up questions the user might want to ask next.",
        min_length=2,
        max_length=3,
    )
    disclaimer: str = Field(
        description="A short disclaimer reminding the user to consult a healthcare professional.",
        default="This information is for educational purposes only. Always consult a qualified healthcare provider.",
    )

    @model_validator(mode="after")
    def check_low_confidence_answer(self) -> "MedicalAnswer":
        if self.confidence == "low":
            warning_phrases = ["not clearly covered", "limited information", "may not be fully"]
            has_warning = any(phrase in self.answer.lower() for phrase in warning_phrases)
            if not has_warning:
                self.answer += " (Note: the source documents have limited information on this topic.)"
        return self

    def to_display(self) -> dict:
        return {
            "answer": self.answer,
            "conditions": self.conditions_mentioned,
            "key_facts": [{"fact": kf.fact, "topic": kf.topic} for kf in self.key_facts],
            "confidence": self.confidence,
            "follow_up": self.follow_up_questions,
            "disclaimer": self.disclaimer,
        }