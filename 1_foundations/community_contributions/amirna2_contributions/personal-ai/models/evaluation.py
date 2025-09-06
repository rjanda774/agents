"""
Evaluation models for the AI Career Assistant.
"""

from typing import List
from pydantic import BaseModel


class Evaluation(BaseModel):
    """Evaluation result for a response"""
    is_acceptable: bool
    feedback: str


class StructuredResponse(BaseModel):
    """Structured response with reasoning and evidence"""
    response: str
    reasoning: str
    tools_used: List[str]
    facts_used: List[str]