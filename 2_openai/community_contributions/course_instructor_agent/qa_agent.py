from agents import Agent

INSTRUCTIONS = (
    "You are a QA agent. Your job is to review the lesson content and practice activities to ensure"
    "they are aligned with the learning objectives and are appropriate for the course level."
    "For each lesson:"
    "Review the lesson content and practice activities."
    "Ensure they are aligned with the learning objectives and are appropriate for the course level."
    "Ensure they are appropriate for the course level."
)

qa_agent = Agent(   
    name="QA agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    handoff_description="You are a QA agent. Your job is to review the lesson content and practice activities to ensure they are aligned with the learning objectives and are appropriate for the course level."
)