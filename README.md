# Stock Picker AI Agent

An intelligent multi-agent system powered by CrewAI that analyzes trending companies in the news and recommends the best investment opportunities based on comprehensive research.

## Overview

This project uses a hierarchical team of AI agents to:
1. **Find trending companies** - Scans latest financial news to identify companies generating buzz in a specific sector
2. **Research companies** - Conducts deep analysis on each trending company's market position, outlook, and investment potential
3. **Pick the best investment** - Synthesizes research findings to recommend the top investment opportunity

## Architecture

The system uses CrewAI's hierarchical process with the following agents:

### Agents

1. **Financial News Analyst** - Finds 2-3 trending companies in the news
2. **Senior Financial Researcher** - Provides comprehensive analysis of each company
3. **Stock Picker** - Selects the best company for investment based on research
4. **Manager** - Orchestrates the entire workflow and delegates tasks

### Features

- **Multi-Agent Collaboration**: Agents work together hierarchically with a manager coordinating tasks
- **Memory System**: 
  - Long-term memory (SQLite) for persistent learning across sessions
  - Short-term memory (RAG) for current context
  - Entity memory for tracking key information
- **Web Search Integration**: Uses SerperDev tool for real-time market research
- **Push Notifications**: Optional Pushover integration to notify you of investment decisions
- **Structured Outputs**: Uses Pydantic models for reliable, typed responses
- **Output Persistence**: Saves reports in JSON and Markdown formats

## Installation

### Prerequisites

- Python >=3.10 <3.13
- [UV package manager](https://docs.astral.sh/uv/)
- OpenAI API key
- Serper API key (for web search)
- (Optional) Pushover account for push notifications

### Setup

1. **Clone this repository:**
```bash
git clone <your-repo-url>
cd stock-picker-isolated
```

2. **Install UV (if not already installed):**
```bash
pip install uv
```

3. **Install CrewAI CLI:**
```bash
uv tool install crewai
```

4. **Install dependencies:**
```bash
crewai install
```

5. **Set up environment variables:**

Create a `.env` file in the root directory:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here

# Optional - for push notifications
PUSHOVER_USER=your_pushover_user_key
PUSHOVER_TOKEN=your_pushover_app_token
```

**Getting API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Serper: https://serper.dev/
- Pushover: https://pushover.net/

## Usage

### Run the Stock Picker

From the root directory:

```bash
crewai run
```

This will:
1. Search for trending companies in the Technology sector (configurable)
2. Research each company's market position and investment potential
3. Select the best company for investment
4. Generate detailed reports in the `output/` directory

### Customization

**Change the sector:**

Edit `src/stock_picker/main.py` and modify the `inputs` dictionary:

```python
inputs = {
    'sector': 'Healthcare',  # Change this to any sector
    "current_date": str(datetime.now())
}
```

**Modify agent behavior:**

- Edit `src/stock_picker/config/agents.yaml` to change agent roles, goals, and backstories
- Edit `src/stock_picker/config/tasks.yaml` to modify task descriptions and outputs

**Change LLM models:**

In `agents.yaml`, you can specify different models:

```yaml
financial_researcher:
  llm: openai/gpt-4o  # Use GPT-4o for more thorough analysis
```

## Project Structure

```
stock-picker/
├── src/stock_picker/
│   ├── main.py                 # Entry point
│   ├── crew.py                 # Agent and task definitions
│   ├── config/
│   │   ├── agents.yaml         # Agent configurations
│   │   └── tasks.yaml          # Task definitions
│   └── tools/
│       ├── __init__.py
│       └── push_tool.py        # Push notification tool
├── output/
│   ├── trending_companies.json # List of trending companies
│   ├── research_report.json    # Detailed research
│   └── decision.md             # Final investment decision
├── memory/                     # Agent memory storage
├── knowledge/                  # User preferences and knowledge
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

## Output Files

The system generates three main outputs:

1. **trending_companies.json** - List of companies trending in the news with their tickers and reasons
2. **research_report.json** - Comprehensive analysis of each company including market position, outlook, and investment potential
3. **decision.md** - Final investment recommendation with detailed rationale

## Memory System

The crew maintains three types of memory:

- **Long-term Memory**: Persists across sessions to remember past analyses and avoid repeating companies
- **Short-term Memory**: Maintains context during the current analysis
- **Entity Memory**: Tracks important information about companies, markets, and trends

Memory files are stored in the `memory/` directory and persist between runs.

## Advanced Usage

### Running Multiple Sectors

You can create a loop to analyze multiple sectors:

```python
sectors = ['Technology', 'Healthcare', 'Finance', 'Energy']
for sector in sectors:
    inputs = {'sector': sector, "current_date": str(datetime.now())}
    result = StockPicker().crew().kickoff(inputs=inputs)
    # Process results...
```

### Integrating with Trading Systems

The output JSON files can be easily integrated with trading APIs or portfolio management systems.

## Troubleshooting

**"ModuleNotFoundError: No module named 'crewai'"**
- Run: `crewai install`

**"API key not found"**
- Ensure your `.env` file is in the root directory with valid API keys

**"SerperDevTool error"**
- Verify your SERPER_API_KEY is correct
- Check your Serper account has available credits

**Memory database locked**
- Delete files in `memory/` directory and restart

## Contributing

This project is part of a larger AI agents learning repository. Feel free to:
- Submit issues for bugs or feature requests
- Create pull requests with improvements
- Share your investment findings (for educational purposes only!)

## Disclaimer

⚠️ **Important**: This tool is for educational and research purposes only. It should NOT be used as the sole basis for investment decisions. Always:
- Conduct your own research
- Consult with financial advisors
- Consider your risk tolerance
- Understand that past performance doesn't guarantee future results

The creators of this tool are not responsible for any financial losses incurred from using this software.

## License

MIT License - See LICENSE file for details

## Credits

Built with:
- [CrewAI](https://crewai.com) - Multi-agent orchestration framework
- [OpenAI](https://openai.com) - Language models
- [Serper](https://serper.dev) - Web search API
- [Pushover](https://pushover.net) - Push notifications

Part of the "Master AI Agentic Engineering" course by Edward Donner.
