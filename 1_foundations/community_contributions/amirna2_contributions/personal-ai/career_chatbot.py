"""
Legacy compatibility wrapper for the original career_chatbot.py

This file maintains backward compatibility with the original monolithic structure
while using the new modular architecture underneath. This ensures that existing
deployments (like Hugging Face) continue to work without any changes.
"""

import logging

# Set up logging at the root level  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import all the original classes from their new modular locations
# This provides 100% backward compatibility
from models import (
    ChatbotConfig,
    Evaluation, 
    StructuredResponse,
    SkillAssessment,
    JobMatchResult
)

from services import (
    NotificationService,
    WebSearchService, 
    DocumentLoader
)

from core import (
    Evaluator,
    ToolRegistry
)

from chatbot import CareerChatbot

# Import the main function from the new main.py
from main import main

# Export all the classes that were previously defined in this file
# This ensures any code that imports from career_chatbot continues to work
__all__ = [
    'ChatbotConfig',
    'Evaluation',
    'StructuredResponse', 
    'SkillAssessment',
    'JobMatchResult',
    'NotificationService',
    'WebSearchService',
    'DocumentLoader',
    'Evaluator',
    'ToolRegistry',
    'CareerChatbot',
    'main'
]

# If this file is run directly, execute the main function
# This maintains the original behavior
if __name__ == "__main__":
    main()
