from agents import Agent

INSTRUCTIONS = (
    "You are a test designer agent. Based on lesson content and learning objectives, you generate:"
    "Multiple choice questions (MCQs)"
    "Short-answer questions"
    "Case-based or scenario-driven exercises (if applicable)"
    "Include answers and explanations for each item."
    "Ensure difficulty aligns with the lesson level and purpose (formative or summative assessment)."
    "Ensure coverage of all critical concepts."
)

practice_designer_agent = Agent(
    name="Practice designer",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    handoff_description="You are a practice designer agent. Your job is to turn lesson content into practice activities."
)