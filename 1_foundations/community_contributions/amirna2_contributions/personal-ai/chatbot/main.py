"""
Main CareerChatbot class for the AI Career Assistant.
"""

import logging
from typing import List, Dict
import gradio as gr
from openai import OpenAI

from models import ChatbotConfig, StructuredResponse
from services import NotificationService, WebSearchService, DocumentLoader
from core import Evaluator, ToolRegistry

logger = logging.getLogger(__name__)


class CareerChatbot:
    """Main chatbot class that orchestrates the AI assistant"""

    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.openai_client = OpenAI()

        # Initialize services
        self.notification_service = NotificationService()
        self.web_search_service = WebSearchService(github_username=config.github_username) if config.github_username else None
        self.document_loader = DocumentLoader()

        # Load professional context
        self.context = self._load_context()

        # Initialize tool registry with context
        self.tool_registry = ToolRegistry(self.notification_service, self.web_search_service,
                                          self.openai_client, self.context, self.config)
        self.evaluator = Evaluator(self.config, self.context)
        self.system_prompt = self._create_system_prompt()

        logger.info(f"CareerChatbot initialized for {config.name}")

    def _load_context(self) -> Dict[str, str]:
        """Load all professional context documents"""
        context = {
            "resume": self.document_loader.load_pdf(self.config.resume_path),
            "linkedin": self.document_loader.load_pdf(self.config.linkedin_path),
            "summary": self.document_loader.load_text(self.config.summary_path)
        }
        return context

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the AI assistant"""
        prompt = f"""You are an AI assistant designed by {self.config.name} and representing them, helping visitors learn about their professional background.
Your knowledge comes from {self.config.name}'s resume, LinkedIn profile, and professional summary provided below.
Your knowledge can also be augmented with real-time data from GitHub if needed and/or when appropriate.

CRITICAL INSTRUCTIONS:
1. ALWAYS search through ALL the provided context (Summary, LinkedIn, Resume) before claiming you don't have information. Be precise and thorough.
2. CONTACT IS A TWO-STEP PROCESS:
   a. First, OFFER to facilitate contact for i) professional questions you can't fully answer, or ii) job matches rated '{self.config.job_match_threshold}' or better. Your response should just be text making the offer.
   b. Second, WAIT for the user to provide their email. ONLY THEN should you use the `record_user_details` tool. Never invent an email.
3. USER-INITIATED CONTACT: If a user asks to connect before you offer, politely decline.
4. PERSONAL QUESTIONS: For private/personal questions (salary, etc.), respond ONLY with "I am sorry, I can't provide that information." and do not offer contact.
5. JOB MATCHING: Use `evaluate_job_match` for job descriptions. Present the full analysis. If the match is good, follow the two-step contact process.
IMPORTANT: The Resume and LinkedIn contain detailed technical information, frameworks, tools, and technologies used. Always check these thoroughly.

TOOLS:
- record_user_details: Record contact information ONLY AFTER a user provides their email.
- evaluate_job_match: Analyze job fit and provide detailed match levels and recommendations

TOOLS:
- record_user_details: Record contact information when someone wants professional follow-up
- evaluate_job_match: Analyze job fit and provide detailed match levels and recommendations"""

        if self.web_search_service:
            prompt += """
- search_github_repos: Search for open source projects and GitHub repositories
- get_repo_details: Get detailed information about specific repositories"""

        prompt += f"""

Be helpful and answer what you know from the context. Use GitHub search tools for questions about open source work, repositories, or recent projects.

## CONTEXT:

### Summary:
{self.context['summary']}

### LinkedIn Profile:
{self.context['linkedin']}

### Resume:
{self.context['resume']}"""

        return prompt

    def chat(self, message: str, history: List[Dict[str, str]], max_retries: int = 3) -> str:
        """Main chat function that processes user messages with evaluation and Lab 3 retry approach"""
        logger.info(f"==> PROCESSING message: '{message[:50]}...'")

        # Generate initial response with tools
        messages = [{"role": "system", "content": self.system_prompt}] + history + [{"role": "user", "content": message}]
        structured_reply, pending_notifications = self._generate_response_with_tools(messages)

        # Safety check - ensure we have a valid structured_reply
        if not structured_reply:
            logger.error("No structured reply received from _generate_response_with_tools")
            return "I apologize, but I'm experiencing technical difficulties. Please try again."

        # For evaluation, use the original history (tool results will be detected from tools_used field)
        evaluation_history = history

        # Systematic evaluation with Lab 3 approach
        for attempt in range(max_retries):
            try:
                # Evaluate the current reply using history that includes tool results
                evaluation = self.evaluator.evaluate_structured(structured_reply, message, evaluation_history)

                if evaluation.is_acceptable:
                    logger.info(f"✅ PASSED evaluation on attempt {attempt + 1}/{max_retries}")

                    # Send notifications only after successful evaluation
                    for notification in pending_notifications:
                        self.tool_registry.notification_service.send(notification)

                    return structured_reply.response if structured_reply else "I apologize, but I'm experiencing technical difficulties."
                else:
                    logger.warning(f"❌ FAILED evaluation on attempt {attempt + 1}/{max_retries}: {evaluation.feedback[:100]}...")

                    # If we haven't exhausted retries, regenerate using Lab 3 rerun approach
                    if attempt < max_retries - 1:
                        logger.info("🔄 Regenerating response with feedback...")
                        # Clear pending notifications from failed attempt
                        pending_notifications.clear()
                        new_reply = self.evaluator.rerun(structured_reply.response, message, history, evaluation.feedback)
                        if new_reply:
                            structured_reply = new_reply
                        else:
                            logger.error("Rerun returned None, keeping original reply")
                    else:
                        logger.warning(f"⚠️ Max retries ({max_retries}) reached. Returning final attempt.")
                        return structured_reply.response if structured_reply else "I apologize, but I'm experiencing technical difficulties."

            except Exception as eval_error:
                logger.error(f"Evaluation failed: {eval_error}")
                # If evaluation fails, return the response we have
                return structured_reply.response if structured_reply else "I apologize, but I'm experiencing technical difficulties."

        return structured_reply.response if structured_reply else "I apologize, but I'm experiencing technical difficulties."

    def _generate_response_with_tools(self, messages: List[Dict]) -> tuple[StructuredResponse, List[str]]:
        """Generate response handling tool calls and collect pending notifications"""
        done = False
        all_pending_notifications = []

        while not done:
            try:
                # Call the LLM with tools and structured output
                response = self.openai_client.beta.chat.completions.parse(
                    model=self.config.model,
                    messages=messages,
                    tools=self.tool_registry.tools,
                    tool_choice="auto",
                    response_format=StructuredResponse
                )
                system_fp = getattr(response, "system_fingerprint", None)
                logging.info("CHAT: served_model=%s system_fp=%s", response.model, system_fp)

                finish_reason = response.choices[0].finish_reason

                # Handle tool calls if needed
                if finish_reason == "tool_calls":
                    message_obj = response.choices[0].message
                    tool_calls = message_obj.tool_calls
                    results, pending_notifications = self.tool_registry.handle_tool_calls(tool_calls)
                    all_pending_notifications.extend(pending_notifications)
                    messages.append(message_obj)
                    messages.extend(results)
                else:
                    done = True
                    return response.choices[0].message.parsed, all_pending_notifications

            except Exception as e:
                logger.error(f"Structured response parsing failed: {e}")
                # Fallback: try without structured output
                try:
                    fallback_response = self.openai_client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        tools=self.tool_registry.tools,
                        tool_choice="auto"
                    )

                    # Create a basic structured response from the fallback
                    content = fallback_response.choices[0].message.content or "I apologize, but I encountered an error processing your request."
                    fallback_structured = StructuredResponse(
                        response=content,
                        reasoning="Fallback response due to parsing error",
                        tools_used=[],
                        facts_used=[]
                    )
                    return fallback_structured, all_pending_notifications

                except Exception as fallback_error:
                    logger.error(f"Fallback response also failed: {fallback_error}")
                    # Ultimate fallback
                    error_response = StructuredResponse(
                        response="I apologize, but I'm experiencing technical difficulties. Please try again.",
                        reasoning="Error handling response",
                        tools_used=[],
                        facts_used=[]
                    )
                    return error_response, all_pending_notifications

    def create_initial_greeting(self) -> str:
        """Create the initial greeting message"""
        return f"""👋 Hello! I'm an AI assistant designed by {self.config.name} and representing them professionally.

I can answer questions about {self.config.name}'s career, experience, and professional background based on their resume and LinkedIn profile.

If you have questions I can't answer from the available information, I'll be happy to arrange for {self.config.name} to respond to you personally via email.

How can I help you today?"""

    def launch_interface(self):
        """Launch the Gradio interface"""
        # Create chatbot with initial message
        chatbot = gr.Chatbot(
            value=[
                {"role": "assistant", "content": self.create_initial_greeting()}
            ],
            type="messages",
            height=700,
            show_copy_button=True,
            show_copy_all_button=True,
        )

        # Create and launch the interface
        interface = gr.ChatInterface(
            self.chat,
            type="messages",
            chatbot=chatbot,
            examples=[
                "What is the professional background?",
                "What companies has this person worked at?",
                "Where did they go to school?",
                "What are their main skills?"
            ],
            title=f"{self.config.name}'s AI Assistant"
        )

        interface.launch()