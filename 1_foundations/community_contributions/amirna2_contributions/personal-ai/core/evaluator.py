"""
Evaluator for AI responses in the Career Assistant.
"""

import os
import logging
from typing import List, Dict
from openai import OpenAI

from models import Evaluation, StructuredResponse, ChatbotConfig

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates AI responses for accuracy and hallucinations"""

    def __init__(self, config: ChatbotConfig, context: Dict[str, str]):
        self.config = config
        self.context = context
        # Use a different model for evaluation to avoid bias
        self.evaluator_client = OpenAI(
                                    api_key=os.getenv("GEMINI_API_KEY"),
                                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                                )

        self.system_prompt = self._create_evaluator_prompt()

    def _create_evaluator_prompt(self) -> str:
        """Create the evaluator system prompt"""

        # Debug logging for evaluator context
        resume_length = len(self.context['resume'])
        linkedin_length = len(self.context['linkedin'])
        summary_length = len(self.context['summary'])
        resume_has_webrtc = "WebRTC" in self.context['resume']
        resume_has_websocket = "WebSocket" in self.context['resume']

        logger.info(f"EVALUATOR CONTEXT DEBUG:")
        logger.info(f"  Resume length: {resume_length} chars, WebRTC: {resume_has_webrtc}, WebSocket: {resume_has_websocket}")
        logger.info(f"  LinkedIn length: {linkedin_length} chars")
        logger.info(f"  Summary length: {summary_length} chars")

        if resume_has_webrtc:
            webrtc_index = self.context['resume'].find("WebRTC")
            snippet = self.context['resume'][max(0, webrtc_index-50):webrtc_index+50]
            logger.info(f"  WebRTC context in resume: ...{snippet}...")

        return f"""You are an intelligent evaluator for an AI agent's structured responses.

The Agent represents {self.config.name} and provides responses in structured format containing:
- response: The actual answer shown to users
- reasoning: How the agent arrived at the answer
- tools_used: List of tools called (if any)
- facts_used: Specific facts/quotes supporting the response

## CONTEXT AVAILABLE TO AGENT:
### Summary:
{self.context['summary']}

### LinkedIn Profile:
{self.context['linkedin']}

### Resume:
{self.context['resume']}

## EVALUATION LOGIC:

### WHEN tools_used is NOT EMPTY:
- Accept tool results (especially GitHub API data) as valid factual information
- Tool results don't need to strictly match resume/LinkedIn context
- GitHub may show languages/technologies or projects not mentioned in resume/LinkedIn - this is VALID
- Verify tool usage was appropriate for the question
- Check that reasoning explains the tool usage

### WHEN tools_used is EMPTY:

- Factual validation: All factual claims must be explicitly supported by resume/summary/LinkedIn context, including technical skills, experiences, tools, and technologies, numbers, dates, and names
- Allow reasonable inferences (e.g., ROS2 experience → DDS knowledge) if clearly explained in reasoning
- Allow some semantic flexibility (e.g. core competencies ↔ core skills) but not major changes
- Reject any information not found in the provided context
- Verify agent follows these behavioral rules:
  1. Professional questions not fully answerable → offers to facilitate contact with {self.config.name}
  2. Personal/private questions (salary, relationships, private details) → MUST respond "I am sorry, I can't provide that information" and MUST NOT offer to facilitate contact
  3. Follow-up requests to contact for personal information → MUST be declined without alternatives
  4. Follow-up requests to contact for job match below threshold → MUST be declined without alternatives
  5. Follow-up requests to contact for professional questions in context → SHOULD facilitate contact and record user details
  6. Job matches at or above threshold ({self.config.job_match_threshold if self.config else 'Good'}) → SHOULD facilitate contact and record user details
  7. HIERARCHY: Very Strong > Strong > Good > Moderate > Weak > Very Weak (Strong is ABOVE Good threshold!)

## DECISION CRITERIA:
- Does the response match the facts_used?
- Is the reasoning sound?
- Were appropriate tools used (or should have been)?
- Are behavioral rules followed?

Mark UNACCEPTABLE only if: unsupported claims, missing tool usage when needed, or behavioral rules violated."""


    def _create_user_prompt(self, reply: str, message: str, history: List[Dict]) -> str:
        """Create the user prompt for evaluation"""
        history_str = "\n".join([f"{h['role']}: {h['content']}" for h in history[-3:]])  # Last 3 messages

        return f"""Here's the conversation context:

{history_str}

Latest User message: {message}

Latest Agent response: {reply}

Please evaluate this response with STRICTNESS:
1. Check EVERY factual claim against the provided context
2. If the Agent mentions ANY specific detail (skills, technologies, experiences, tools) not explicitly in the context, mark as UNACCEPTABLE
3. If the Agent should have said "I don't have that information", but instead made something up, mark as UNACCEPTABLE
4. Look for common hallucinations like claiming experience with technologies not mentioned in the resume/LinkedIn

Is this response acceptable? Provide specific feedback about any issues."""

    def rerun(self, reply: str, message: str, history: List[Dict], feedback: str) -> StructuredResponse:
        """Regenerate structured response with feedback from failed evaluation"""
        updated_system_prompt = self._create_base_system_prompt() + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
        updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
        updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
        updated_system_prompt += "Please provide a corrected structured response that addresses the feedback."

        messages = [{"role": "system", "content": updated_system_prompt}] + history + [{"role": "user", "content": message}]

        response = self.evaluator_client.beta.chat.completions.parse(
            model=self.config.evaluator_model,
            messages=messages,
            response_format=StructuredResponse
        )
        system_fp = getattr(response, "system_fingerprint", None)
        logging.info("EVAL: served_model=%s system_fp=%s", response.model, system_fp)

        return response.choices[0].message.parsed

    def _create_base_system_prompt(self) -> str:
        """Create base system prompt without evaluation context"""
        prompt = f"""You are an AI assistant representing {self.config.name}, helping visitors learn about their professional background.

Your knowledge comes from {self.config.name}'s resume, LinkedIn profile, and professional summary provided below.

CRITICAL INSTRUCTIONS:
1. ALWAYS search through ALL the provided context (Summary, LinkedIn, Resume) before claiming you don't have information. Be precise and thorough.
2. Only say "I don't have that information" if you've thoroughly searched and the information is genuinely NOT in any of the provided documents.
3. For professional questions not fully covered in context, offer to facilitate contact with {self.config.name}.
4. For personal/private information (salary, relationships, private details), simply say: "I am sorry, I can't provide that information." DO NOT offer to facilitate contact for personal questions.

IMPORTANT: The Resume and LinkedIn contain detailed technical information, frameworks, tools, and technologies used. Always check these thoroughly.

TOOLS:
- record_unknown_question: Record professional questions you cannot answer from the context
- record_user_details: Record contact information when someone wants professional follow-up

Be helpful and answer what you know from the context."""

        prompt += f"""

## CONTEXT:

### Summary:
{self.context['summary']}

### LinkedIn Profile:
{self.context['linkedin']}

### Resume:
{self.context['resume']}"""

        return prompt

    def evaluate_structured(self, structured_reply: StructuredResponse, message: str, history: List[Dict]) -> Evaluation:
        """Evaluate a structured response with reasoning and evidence"""
        try:
            # Create enhanced user prompt that includes the structured information
            is_job_matching = self._is_job_matching_context(structured_reply, message, history)

            # Check if GitHub tools were used
            github_tools_used = any(tool in structured_reply.tools_used for tool in ['search_github_repos', 'get_repo_details'])

            if is_job_matching:
                evaluation_criteria = f"""Please evaluate this job matching response with REASONABLE STANDARDS:
1. Is the reasoning sound for professional skill assessment?
2. Are technical inferences reasonable (e.g., ROS2 experience → DDS knowledge)?
3. Were appropriate tools used for job analysis?
4. Does the response provide useful insights for recruitment?
5. CRITICAL: Match level hierarchy is Very Strong > Strong > Good > Moderate > Weak > Very Weak
6. CRITICAL: If job match is "{self.config.job_match_threshold if self.config else 'Good'}" or HIGHER in the hierarchy (Strong, Very Strong), facilitating contact is CORRECT behavior
7. CRITICAL: If job match is LOWER in the hierarchy than "{self.config.job_match_threshold if self.config else 'Good'}" (Moderate, Weak, Very Weak), declining contact is CORRECT behavior

Job matching responses should be evaluated for practical utility, not pedantic precision.
Accept reasonable technical inferences and contact facilitation decisions based on match level."""
            else:
                if github_tools_used:
                    evaluation_criteria = """Please evaluate this response with REASONABLE STANDARDS for GitHub tool usage:
1. GitHub tools (search_github_repos, get_repo_details) were used to gather additional information
2. Repository details like stars, forks, creation dates, programming languages, topics are LEGITIMATE from GitHub API
3. Technical project details obtained from GitHub tools are acceptable
4. Only reject if claims obviously contradict the professional background
5. The agent appropriately used tools to provide detailed project information

When GitHub tools are used, trust the additional technical details they provide.
Is this response acceptable? Provide specific feedback about any issues."""
                else:
                    evaluation_criteria = """Please evaluate this response with STRICTNESS:
1. Check EVERY factual claim against the provided context
2. If the Agent mentions ANY specific detail not explicitly in the context, mark as UNACCEPTABLE
3. If the Agent should have said "I don't have that information", but instead made something up, mark as UNACCEPTABLE
4. Look for common hallucinations and unsupported claims

Is this response acceptable? Provide specific feedback about any issues."""

            newline = '\n'
            user_prompt = f"""Here's the conversation context:

{newline.join([f"{h['role']}: {h['content']}" for h in history[-3:]])}

Latest User message: {message}

Agent's structured response:
Response: {structured_reply.response}
Reasoning: {structured_reply.reasoning}
Tools used: {structured_reply.tools_used}
Facts used: {structured_reply.facts_used}

{evaluation_criteria}"""

            # Check if GitHub tool results should be included in system prompt
            github_context = self._extract_github_context_from_history(history)
            system_prompt = self._create_evaluator_prompt_with_github(github_context) if github_context else self._create_evaluator_prompt()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = self.evaluator_client.beta.chat.completions.parse(
                model=self.config.evaluator_model,
                messages=messages,
                response_format=Evaluation,
                temperature=0.0
            )
            system_fp = getattr(response, "system_fingerprint", None)
            logging.info("EVAL: served_model=%s system_fp=%s", response.model, system_fp)

            evaluation = response.choices[0].message.parsed
            logger.info(f"EVALUATION RESULT: {'PASS' if evaluation.is_acceptable else 'FAIL'}")
            logger.info(f"AGENT RESPONSE: {structured_reply.response}")
            logger.info(f"AGENT REASONING: {structured_reply.reasoning}")
            logger.info(f"EVALUATOR FEEDBACK: {evaluation.feedback}")
            return evaluation

        except Exception as e:
            logger.error(f"Structured evaluation failed: {e}")
            return Evaluation(is_acceptable=True, feedback=f"Evaluation error: {str(e)}")

    def _external_tools_used(self, history: List[Dict]) -> bool:
        """Check if tools with external data (GitHub, job matching) were used in the conversation"""
        for message in history:
            if message.get('role') == 'tool':
                content = message.get('content', '')
                print(f"TOOL CONTENT DEBUG: {content}")  # Temporary debug
                # Check for GitHub tool results
                if any(indicator in content for indicator in ['repos', 'languages_found', 'total_repos', 'github.com']):
                    return True
                # Check for job matching tool results
                if any(indicator in content for indicator in ['overall_match_level', 'skill_assessments', 'should_facilitate_contact']):
                    print("JOB MATCHING TOOL DETECTED!")  # Temporary debug
                    return True
        print("NO EXTERNAL TOOLS DETECTED")  # Temporary debug
        return False

    def _is_github_context(self, structured_reply: StructuredResponse) -> bool:
        """Check if GitHub tools were used"""
        return any(tool in structured_reply.tools_used for tool in ['search_github_repos', 'get_repo_details'])

    def _is_job_matching_context(self, structured_reply: StructuredResponse, message: str, history: List[Dict]) -> bool:
        """Check if this is a job matching context"""
        # Check if job matching tool was used
        if 'evaluate_job_match' in structured_reply.tools_used:
            return True

        # Check if response contains job matching indicators
        response_content = structured_reply.response.lower()
        if any(indicator in response_content for indicator in ['match level', 'skills breakdown', 'overall match', 'job fit']):
            return True

        # Check if message contains job posting indicators
        message_content = message.lower()
        if any(indicator in message_content for indicator in ['job description', 'role', 'position', 'hiring', 'candidate']):
            return True

        return False

    def _create_evaluator_prompt_with_github(self, github_context: str) -> str:
        """Create evaluator prompt including GitHub tool results as valid context"""
        base_prompt = self._create_evaluator_prompt()

        if github_context:
            enhanced_prompt = base_prompt.replace(
                "Mark UNACCEPTABLE only if: unsupported claims, missing tool usage when needed, or behavioral rules violated.",
                f"""## GitHub Tool Results (VALID CONTEXT):
{github_context}

IMPORTANT: GitHub tool results above are LEGITIMATE CONTEXT.

When evaluating responses about programming languages, repositories, or technical projects:
1. GitHub tool results are VALID and should be considered alongside resume/LinkedIn
2. Programming languages found in GitHub repos are FACTUAL, not hallucinations
3. The agent should synthesize information from resume/LinkedIn AND GitHub tool results
4. Only reject if GitHub results contradict the general technical background in resume/LinkedIn

Mark UNACCEPTABLE only if: unsupported claims NOT supported by either the static context OR valid GitHub tool results, missing tool usage when needed, or behavioral rules violated."""
            )
            return enhanced_prompt

        return base_prompt

    def _extract_github_context_from_history(self, history: List[Dict]) -> str:
        """Extract GitHub tool results from conversation history"""
        github_context = ""

        for message in history:
            if message.get('role') == 'tool':
                content = message.get('content', '')
                # Check if this is GitHub tool content (repo details or repo search results)
                if any(indicator in content for indicator in [
                    'full_name', 'html_url', 'stargazers_count', 'watchers_count', 'forks_count',
                    'open_issues_count', 'created_at', 'updated_at', 'topics', 'repos', 'github.com'
                ]):
                    github_context += f"\n{content}"

        return github_context.strip()