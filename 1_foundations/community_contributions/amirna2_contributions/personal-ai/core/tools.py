"""
Tool registry for AI agents in the Career Assistant.
"""

import json
import logging
from typing import List, Dict, Optional
from openai import OpenAI

from models import JobMatchResult, ChatbotConfig
from services import NotificationService, WebSearchService

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Manages AI agent tools and their execution"""

    def __init__(self, notification_service: NotificationService, web_search_service: Optional[WebSearchService] = None,
                 openai_client: Optional[OpenAI] = None, context: Optional[Dict[str, str]] = None,
                 config: Optional[ChatbotConfig] = None):
        self.notification_service = notification_service
        self.web_search_service = web_search_service
        self.openai_client = openai_client
        self.context = context or {}
        self.config = config
        self.tools = self._create_tool_definitions()

    def _create_tool_definitions(self) -> List[Dict]:
        """Create tool definitions for the AI agent"""
        record_user_details = {
            "name": "record_user_details",
            "strict": True,
            "description": (
                "Use this tool ONLY AFTER a user has explicitly provided their email address in response to an offer to facilitate contact. "
                "This tool records the user's contact details. "
                "IMPORTANT: DO NOT use this tool unless the user has given you their email. Do not make up an email address."
            ),
            "parameters": {
                "type": "object",
                "strict": True,
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The email address explicitly provided by the user. Do not invent this."
                    },
                    "name": {
                        "type": "string",
                        "description": "The user's name if they provided it. If not, use 'Visitor'."
                    },
                    "notes": {
                        "type": "string",
                        "description": (
                            "Detailed notes about the conversation context and why the user wants to be contacted. "
                            "Include the original question or job match details."
                        )
                    }
                },
                "required": ["email", "name", "notes"],
                "additionalProperties": False
            }
        }


        evaluate_job_match = {
            "name": "evaluate_job_match",
            "strict": True,
            "description": (
                "Analyze how well the candidate matches a job posting. Use this when someone asks "
                "about job fit, role suitability, or provides a job description to evaluate. "
                "Returns detailed analysis with match levels and recommendations."
            ),
            "parameters": {
                "type": "object",
                "strict": True,
                "properties": {
                    "job_description": {
                        "type": "string",
                        "description": "The FULL, COMPLETE, UNEDITED job description text exactly as provided by the user. Do NOT summarize, extract, or truncate - include ALL details including company info, salary, responsibilities, requirements, and nice-to-haves."
                    },
                    "role_title": {
                        "type": "string",
                        "description": "The job title or role name"
                    }
                },
                "required": ["job_description", "role_title"],
                "additionalProperties": False
            }
        }

        tools = [
            {"type": "function", "function": record_user_details},
            {"type": "function", "function": evaluate_job_match}
        ]

        # Add GitHub search tools if web search service is available
        if self.web_search_service:
            search_github = {
                "name": "search_github_repos",
                "strict": True,
                "description": (
                    "Get ALL GitHub repositories with full details including languages, topics, stars, etc. "
                    "Call WITHOUT parameters to get everything, then analyze the returned data. "
                    "Returns list of all repos with language field showing what each is written in."
                ),
                "parameters": {
                    "type": "object",
                    "strict": True,
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }

            get_repo_info = {
                "name": "get_repo_details",
                "strict": True,
                "description": "Get detailed information about a specific GitHub repository",
                "parameters": {
                    "type": "object",
                    "strict": True,
                    "properties": {
                        "repo_name": {
                            "type": "string",
                            "description": "The name of the repository to get details for"
                        }
                    },
                    "required": ["repo_name"],
                    "additionalProperties": False
                }
            }

            tools.extend([
                {"type": "function", "function": search_github},
                {"type": "function", "function": get_repo_info}
            ])

        return tools

    def record_user_details(self, email: str, name: str = "Visitor", notes: str = "not provided") -> Dict:
        """Record user contact details and prepare notification"""
        message = f"Recording interest from {name} with email {email} and notes: {notes}"
        logger.info(f"Recorded user details: {email}, {name}")
        return {
            "recorded": "ok",
            "pending_notification": message
        }


    def evaluate_job_match(self, job_description: str, role_title: str) -> Dict:
        """Evaluate how well the candidate matches a job using LLM analysis"""
        if not self.openai_client or not self.context:
            return {"error": "Job matching requires OpenAI client and context"}

        # Create analysis prompt
        analysis_prompt = f"""You are a professional job matching analyst. Analyze how well this candidate matches the given job.

JOB TITLE: {role_title}
JOB DESCRIPTION: {job_description}

CANDIDATE BACKGROUND:
Summary: {self.context.get('summary', 'Not available')}
Resume: {self.context.get('resume', 'Not available')}
LinkedIn: {self.context.get('linkedin', 'Not available')}

CRITICAL INSTRUCTIONS:
- Only analyze skills and technologies EXPLICITLY mentioned in the job description above
- Do not infer, assume, or add skills that are not directly stated in the job requirements
- Do not include general software engineering practices unless specifically mentioned in the job

Provide a detailed analysis with:
1. Overall match level: Your holistic judgment using EXACTLY one of these levels (you must use these EXACT words only):
   - "Very Strong": 90%+ of skills Extensive/Solid, minimal gaps
   - "Strong": 70-89% of skills Extensive/Solid, few gaps
   - "Good": 50-69% of skills Extensive/Solid/Moderate, manageable gaps
   - "Moderate": 30-49% of skills covered, significant gaps but some foundation
   - "Weak": 10-29% of skills covered, majority missing/limited
   - "Very Weak": <10% of skills covered, complete domain mismatch

   CALIBRATION: Count your skill assessments and calculate the percentage that are Extensive/Solid/Moderate vs Missing/Limited/Inferred. Use this to determine the correct level.

   CRITICAL: Use ONLY these exact 6 levels. Do NOT use "Low", "High", "Fair", "Poor" or any other terms.
2. Skill assessments: For each skill mentioned in the job description, assess using these levels:
   - "Extensive": Multiple projects/companies, clearly a core competency
   - "Solid": Several projects, reliable experience
   - "Moderate": Some mention, decent experience
   - "Limited": Minimal mention or recent/brief exposure
   - "Inferred": Not explicitly mentioned but has closely related/transferable skills (e.g., has MQTT or ROS2 experience for DDS requirement)
   - "Missing": No evidence and no related transferable skills
   - Evidence: Where skill was found OR reasoning for inference/missing assessment
3. Skill assessments format: ALWAYS use the format:
   - Skill Name: Level - Evidence/Reasoning
   - Example: "UI/UX Design: Limited - Some involvement in UI bug fixes but not a core focus in his career."
4. Experience analysis: How candidate's experience aligns with role requirements
5. Industry analysis: How candidate's industry background fits
6. Recommendations: Overall assessment and next steps

CRITICAL: Contact facilitation for jobs must be based STRICTLY on overall match level:
- If match level is "{self.config.job_match_threshold if self.config else 'Good'}" or better: Set should_facilitate_contact = true and offer to facilitate contact
- If match level is below "{self.config.job_match_threshold if self.config else 'Good'}": Set should_facilitate_contact = false and do NOT offer contact facilitation

The hierarchy is: Very Strong > Strong > Good > Moderate > Weak > Very Weak
This threshold is ABSOLUTE - NO exceptions.
"""

        try:
            response = self.openai_client.beta.chat.completions.parse(
                model=self.config.job_matching_model if self.config else "gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional job matching analyst."},
                    {"role": "user", "content": analysis_prompt}
                ],
                response_format=JobMatchResult
            )
            system_fp = getattr(response, "system_fingerprint", None)
            logging.info("MATCH: served_model=%s system_fp=%s", response.model, system_fp)

            result = response.choices[0].message.parsed
            logger.info(f"Job match analysis completed: {result.overall_match_level} match for {role_title}")

            result_dict = result.model_dump()

            # Add pending notification for high matches
            if result.should_facilitate_contact:
                result_dict["pending_notification"] = f"High job match found ({result.overall_match_level}): {role_title}"

            return result_dict

        except Exception as e:
            logger.error(f"Job matching analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def handle_tool_calls(self, tool_calls) -> tuple[List[Dict], List[str]]:
        """Execute tool calls from the AI agent and collect pending notifications"""
        results = []
        pending_notifications = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            logger.info(f"Tool called: {tool_name} with args: {arguments}")

            # Execute the appropriate tool
            if tool_name == "record_user_details":
                result = self.record_user_details(**arguments)
            elif tool_name == "search_github_repos" and self.web_search_service:
                topic = arguments.get('topic')
                result = self.web_search_service.search_github_repos(topic=topic)
            elif tool_name == "get_repo_details" and self.web_search_service:
                repo_name = arguments.get('repo_name')
                result = self.web_search_service.get_repo_details(repo_name)
            elif tool_name == "evaluate_job_match":
                result = self.evaluate_job_match(**arguments)
            else:
                logger.warning(f"Unknown tool called: {tool_name}")
                result = {}

            # Extract pending notifications
            if isinstance(result, dict) and "pending_notification" in result:
                pending_notifications.append(result.pop("pending_notification"))

            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })

        return results, pending_notifications