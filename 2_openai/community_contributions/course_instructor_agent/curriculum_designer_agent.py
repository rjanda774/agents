from pydantic import BaseModel, Field

from agents import Agent

INSTRUCTIONS = (
    "You are a curriculum designer. Given a course outline, you design a curriculum for the course. "
    "Break it into logical modules and lessons."
    "Define learning objectives for each module."
    "Recommend prerequisites and progression flow."
    "The curriculum should be a list of modules, each with a title, description, and a list of lessons. "
    "Each lesson should have a title, description, and a list of activities. "
    "Each activity should have a title, description, and a list of resources. "
)

class Activity(BaseModel):
    """
    Represents a learning activity within a lesson.
    
    Attributes:
        title (str): The name or heading of the activity
        description (str): Detailed explanation of what the activity entails
    """
    title: str = Field(description="The title of the activity")
    description: str = Field(description="The description of the activity")
    
class Lesson(BaseModel):
    """
    Represents a lesson within a module of the curriculum.
    
    Attributes:
        title (str): The name or heading of the lesson
        description (str): Detailed explanation of the lesson content and objectives
        activities (list[Activity]): Collection of learning activities that make up the lesson
    """
    title: str = Field(description="The title of the lesson")
    description: str = Field(description="The description of the lesson")
    activities: list[Activity] = Field(description="The list of activities in the lesson")
    
class Module(BaseModel):
    """
    Represents a module, which is a major section of the curriculum containing multiple lessons.
    
    Attributes:
        title (str): The name or heading of the module
        description (str): Detailed explanation of the module's content and learning objectives
        lessons (list[Lesson]): Collection of lessons that make up the module
    """
    title: str = Field(description="The title of the module")
    description: str = Field(description="The description of the module")
    lessons: list[Lesson] = Field(description="The list of lessons in the module")
    
class Curriculum(BaseModel):
    """
    Represents the complete curriculum structure for a course.
    
    Attributes:
        modules (list[Module]): Collection of all modules that make up the complete curriculum
    """
    modules: list[Module] = Field(description="The list of the modules of the course")

curriculum_designer_agent = Agent(
    name="Curriculum designer",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    output_type=Curriculum,
    handoff_description="You are a curriculum designer. Given a course outline, you design a curriculum for the course. "
)