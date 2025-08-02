from pydantic import BaseModel, Field

from agents import Agent

INSTRUCTIONS = (
    "You are an instructional designer agent. Your job is to turn curriculum outlines into engaging, clear, and pedagogically sound lesson content."
    "For each lesson:"
    "Follow the learning objectives."
    "Write explanations, examples, and step-by-step breakdowns."
    "Incorporate analogies or visuals when appropriate (text description only)."
    "Keep tone aligned with the course level (beginner, intermediate, expert)."
    "Do not generate quizzes or assessments. Pass structured lessons to the next agent."
)

class Instruction(BaseModel):
    title: str = Field(description="The title of the instruction")
    description: str = Field(description="The description of the instruction")


instruction_designer_agent = Agent(
    name="Instruction designer",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    handoff_description="You are a course instructor. Given a course outline, you design instructions for the course. "
)

