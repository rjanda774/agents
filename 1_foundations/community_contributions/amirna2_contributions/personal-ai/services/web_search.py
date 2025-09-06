"""
Web search service for the AI Career Assistant.
"""

import os
import logging
from typing import Optional, Dict, Any
import requests

logger = logging.getLogger(__name__)


class WebSearchService:
    """Handles web searches and GitHub repository lookups"""

    def __init__(self, github_username: Optional[str] = None):
        self.github_username = github_username
        self.github_api_base = "https://api.github.com"
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'CareerChatbot/1.0'
        })

        # Check if GitHub token is available for higher rate limits
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            self.session.headers.update({'Authorization': f'token {github_token}'})
            logger.info("GitHub API configured with authentication")
        else:
            logger.info("GitHub API configured without authentication (rate limits apply)")

    def search_github_repos(self, username: Optional[str] = None, topic: Optional[str] = None) -> Dict[str, Any]:
        """Search GitHub repositories for a user - returns ALL repos with full details"""
        try:
            username = username or self.github_username
            if not username:
                return {"error": "No GitHub username provided", "repos": []}

            # Get user's repositories
            url = f"{self.github_api_base}/users/{username}/repos"
            params = {'sort': 'updated', 'per_page': 100}  # 100 is probably overkill but just in case

            response = self.session.get(url, params=params)
            response.raise_for_status()

            repos = response.json()

            # Filter out forked repositories to show only original work
            repos = [repo for repo in repos if not repo.get('fork', False)]

            # If topic is provided and valid, try to filter (but handle bad inputs gracefully)
            if topic and isinstance(topic, str):
                topic_lower = topic.lower()
                filtered = []
                for repo in repos:
                    # Check topics
                    if any(topic_lower in topic.lower() for topic in repo.get('topics', [])):
                        filtered.append(repo)
                        continue
                    # Check description
                    description = repo.get('description', '') or ''
                    if topic_lower in description.lower():
                        filtered.append(repo)
                        continue
                    # Check name
                    name = repo.get('name', '') or ''
                    if topic_lower in name.lower():
                        filtered.append(repo)
                        continue
                    # Check language
                    language = repo.get('language', '') or ''
                    if topic_lower == language.lower():
                        filtered.append(repo)

                # Only use filtered results if we found matches
                if filtered:
                    repos = filtered

            # Format ALL repos with comprehensive details
            formatted_repos = []
            all_languages = set()

            for repo in repos:  # Return ALL repos, not just 5
                language = repo.get('language')
                if language:
                    all_languages.add(language)

                formatted_repos.append({
                    'name': repo.get('name'),
                    'description': repo.get('description', 'No description'),
                    'url': repo.get('html_url'),
                    'language': language or 'Not specified',
                    'stars': repo.get('stargazers_count', 0),
                    'forks': repo.get('forks_count', 0),
                    'updated': repo.get('updated_at', ''),
                    'created': repo.get('created_at', ''),
                    'topics': repo.get('topics', []),
                    'size': repo.get('size', 0),
                    'is_fork': repo.get('fork', False),
                    'archived': repo.get('archived', False)
                })

            return {
                "username": username,
                "total_repos": len(formatted_repos),
                "languages_used": list(all_languages),
                "topic_searched": topic,
                "repos": formatted_repos
            }

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return {"error": f"GitHub user '{username}' not found", "repos": []}
            else:
                logger.error(f"GitHub API error: {e}")
                return {"error": f"GitHub API error: {str(e)}", "repos": []}
        except Exception as e:
            logger.error(f"Error searching GitHub: {e}")
            return {"error": f"Error searching GitHub: {str(e)}", "repos": []}

    def get_repo_details(self, repo_name: str, username: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a specific repository"""
        try:
            username = username or self.github_username
            if not username:
                return {"error": "No GitHub username provided"}

            url = f"{self.github_api_base}/repos/{username}/{repo_name}"
            response = self.session.get(url)
            response.raise_for_status()

            repo = response.json()

            # Get README content if available
            readme_content = None
            try:
                readme_url = f"{self.github_api_base}/repos/{username}/{repo_name}/readme"
                readme_response = self.session.get(readme_url)
                if readme_response.status_code == 200:
                    readme_data = readme_response.json()
                    if 'content' in readme_data:
                        import base64
                        readme_content = base64.b64decode(readme_data['content']).decode('utf-8')[:500]  # First 500 chars
            except Exception as e:
                logger.debug(f"Could not retrieve README: {e}")
                # Don't let README failure break the entire tool
                pass

            return {
                'name': repo.get('name'),
                'full_name': repo.get('full_name'),
                'description': repo.get('description'),
                'url': repo.get('html_url'),
                'homepage': repo.get('homepage'),
                'language': repo.get('language'),
                'languages_url': repo.get('languages_url'),
                'created_at': repo.get('created_at'),
                'updated_at': repo.get('updated_at'),
                'pushed_at': repo.get('pushed_at'),
                'size': repo.get('size'),
                'stars': repo.get('stargazers_count'),
                'watchers': repo.get('watchers_count'),
                'forks': repo.get('forks_count'),
                'open_issues': repo.get('open_issues_count'),
                'topics': repo.get('topics', []),
                'readme_preview': readme_content
            }

        except Exception as e:
            logger.error(f"Error getting repo details: {e}")
            return {"error": f"Error getting repository details: {str(e)}"}