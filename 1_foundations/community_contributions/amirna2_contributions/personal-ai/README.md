# AI Career Assistant

An AI-powered career assistant that represents professionals on their websites, answering questions about their background while facilitating follow-up contact for qualified opportunities. Built with a modern modular architecture using OpenAI's latest structured output features.

## Features

- **Intelligent Q&A**: Answers questions about professional background using resume, LinkedIn, and summary documents
- **GitHub Integration**: Real-time repository analysis and project showcasing
- **Job Matching**: Advanced LLM-powered job fit analysis with detailed skill assessments
- **Contact Facilitation**: Smart contact routing based on query type and job match quality
- **Response Evaluation**: Built-in quality control system to prevent hallucinations
- **Push Notifications**: Pushover integration for real-time alerts
- **Web Interface**: Clean Gradio-based chat interface

## Architecture

This project follows a clean modular architecture with clear separation of concerns:

```
personal-ai/
├── models/              # Data models & schemas
│   ├── config.py           # Configuration classes
│   ├── evaluation.py       # Response evaluation models
│   └── job_matching.py     # Job analysis models
├── services/            # External service integrations
│   ├── notification.py     # Pushover notifications
│   ├── web_search.py       # GitHub API integration
│   └── document_loader.py  # PDF/text processing
├── core/               # Business logic
│   ├── evaluator.py        # Response quality control
│   └── tools.py            # AI agent tools registry
├── chatbot/            # Main application
│   └── main.py             # CareerChatbot orchestration
├── main.py             # Application entry point
└── career_chatbot.py   # Legacy compatibility wrapper
```

## Installation

### Option 1: Using uv (Recommended)

1. **Install uv (if not already installed):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or with pip: pip install uv
   ```

2. **Clone and navigate to the project:**
   ```bash
   cd personal-ai
   ```

3. **Create virtual environment and install dependencies:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   
   # Alternative: Install using pyproject.toml
   # uv pip install -e .
   ```

### Option 2: Using pip (Traditional)

1. **Clone and navigate to the project:**
   ```bash
   cd personal-ai
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the parent directory with:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key  # For evaluation
   GITHUB_USERNAME=your_github_username  # Optional
   GITHUB_TOKEN=your_github_token        # Optional, for higher rate limits
   PUSHOVER_USER=your_pushover_user      # Optional
   PUSHOVER_TOKEN=your_pushover_token    # Optional
   ```

5. **Prepare your documents:**
   Place your professional documents in the `me/` directory:
   - `resume.pdf` - Your resume
   - `linkedin.pdf` - LinkedIn profile export
   - `summary.txt` - Professional summary

## Usage

### Basic Usage
```bash
python main.py
```

### Legacy Compatibility
```bash
python career_chatbot.py
```

### Programmatic Usage
```python
from models import ChatbotConfig
from chatbot import CareerChatbot

config = ChatbotConfig(
    name="Your Name",
    github_username="your_username"
)

chatbot = CareerChatbot(config)
chatbot.launch_interface()
```

## Configuration

The `ChatbotConfig` class supports extensive customization:

```python
config = ChatbotConfig(
    name="Professional Name",
    github_username="github_user",
    resume_path="me/resume.pdf",
    linkedin_path="me/linkedin.pdf",
    summary_path="me/summary.txt",
    model="gpt-4o-mini-2024-07-18",
    evaluator_model="gemini-2.5-flash",
    job_matching_model="gpt-4o-2024-08-06",
    job_match_threshold="Good"
)
```

## AI Agent Tools

The system includes several specialized tools:

- **`record_user_details`**: Captures contact information for follow-up
- **`evaluate_job_match`**: Analyzes job fit using advanced LLM reasoning
- **`search_github_repos`**: Retrieves and analyzes GitHub repositories
- **`get_repo_details`**: Provides detailed repository information

## Job Matching

The job matching system uses a sophisticated 6-level hierarchy:

- **Very Strong** (90%+ skills): Minimal gaps, excellent fit
- **Strong** (70-89% skills): Few gaps, strong candidate
- **Good** (50-69% skills): Manageable gaps, solid fit
- **Moderate** (30-49% skills): Significant gaps, some foundation
- **Weak** (10-29% skills): Major gaps, limited relevance
- **Very Weak** (<10% skills): Complete domain mismatch

## Quality Control

Built-in evaluation system prevents hallucinations:

- **Factual Validation**: All claims verified against source documents
- **Tool Usage Verification**: Ensures appropriate tool selection
- **Behavioral Rules**: Enforces proper contact facilitation logic
- **Retry Mechanism**: Automatically regenerates poor responses

## Deployment

### Hugging Face Spaces

The legacy wrapper ensures seamless deployment to Hugging Face Spaces without any configuration changes.

### Local Development

**With uv (Recommended):**
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies (choose one)
uv pip install -r requirements.txt
# OR install with pyproject.toml including dev dependencies
uv pip install -e ".[dev]"

# Run the application
python main.py

# Optional: Run with development tools
ruff check .  # Linting
black .       # Code formatting
```

**With pip:**
```bash
# Install in development mode
pip install -e .

# Run with hot reload (if using Gradio's reload feature)
python main.py
```

## Example Interactions

**Professional Question:**
> "What experience does this person have with robotics?"

**Job Matching:**
> "Here's a Senior Robotics Engineer position at Boston Dynamics. How well would this person fit?"

**GitHub Projects:**
> "Can you show me some of their open source work?"

## Migration from Monolith

If upgrading from the original monolithic version:

1. The modular version is fully backward compatible
2. All existing imports continue to work
3. No deployment configuration changes needed
4. Original `career_chatbot.py` now acts as a compatibility wrapper

## Testing

```bash
# Test basic imports
python -c "import career_chatbot; print('Legacy compatibility works')"

# Test modular imports
python -c "from chatbot import CareerChatbot; print('Modular structure works')"
```

## License

This project is part of the Agentic AI Engineering Course community contributions.

## Contributing

This is a community contribution demonstrating modular architecture principles for AI applications. Feel free to use this structure as a template for your own AI assistant projects.

## Contact

This AI assistant was built by Amir Nathoo as a practical demonstration of modular AI application architecture using LLMs and agentic systems.
