"""
Notification service for the AI Career Assistant.
"""

import os
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)


class NotificationService:
    """Handles push notifications via Pushover"""

    def __init__(self, user_token: Optional[str] = None, app_token: Optional[str] = None):
        self.user_token = user_token or os.getenv("PUSHOVER_USER")
        self.app_token = app_token or os.getenv("PUSHOVER_TOKEN")
        self.api_url = "https://api.pushover.net/1/messages.json"
        self.enabled = bool(self.user_token and self.app_token)

        if self.enabled:
            logger.info("Pushover notification service initialized")
        else:
            logger.warning("Pushover credentials not found - notifications disabled")

    def send(self, message: str) -> bool:
        """Send a push notification"""
        if not self.enabled:
            logger.info(f"Notification (disabled): {message}")
            return False

        try:
            payload = {
                "user": self.user_token,
                "token": self.app_token,
                "message": message
            }
            response = requests.post(self.api_url, data=payload)
            response.raise_for_status()
            logger.info(f"Notification sent: {message}")
            return True
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False