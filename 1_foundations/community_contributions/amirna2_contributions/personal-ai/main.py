"""
Main entry point for the AI Career Assistant.
"""

import os
import logging
from dotenv import load_dotenv

from models import ChatbotConfig
from chatbot import CareerChatbot

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the application"""
    # Load environment variables
    load_dotenv(dotenv_path="../.env", override=True)

    # Create configuration
    # Extract GitHub username from summary or environment variable
    github_username = os.getenv("GITHUB_USERNAME")  # Can be set to actual username
    config = ChatbotConfig(
        name="Amir Nathoo",
        github_username=github_username  # Set to actual GitHub username if available
    )

    # Initialize and launch chatbot
    chatbot = CareerChatbot(config)
    chatbot.launch_interface()


if __name__ == "__main__":
    main()
