"""
Data models for the AI Career Assistant.
"""

from .config import ChatbotConfig
from .evaluation import Evaluation, StructuredResponse
from .job_matching import SkillAssessment, JobMatchResult

__all__ = [
    'ChatbotConfig',
    'Evaluation', 
    'StructuredResponse',
    'SkillAssessment',
    'JobMatchResult'
]